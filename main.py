"""
Real-time face recognition — zero-lag, flicker-free.

Architecture:
  • Main thread  : captures frames at full camera FPS, renders overlay, displays.
  • Worker thread: runs DeepFace.find() on a snapshot; updates results
                   only when the previous recognition pass is complete.

This keeps the display loop free from any blocking calls.
"""

import threading
import time
from pathlib import Path

import cv2
_GUI_OK = True
import numpy as np
from deepface import DeepFace

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH          = Path(__file__).parent / "face_database"
MODEL_NAME       = "ArcFace"
DETECTOR_BACKEND = "retinaface"
CAMERA_INDEX     = 0
WINDOW_TITLE     = "Real-time Face Recognition  |  Q to quit"

# Colours (BGR)
COL_MATCH    = (0,   220,  80)   # bright green  – known face
COL_UNKNOWN  = (0,   140, 255)   # amber         – face detected, no match
COL_TEXT_BG  = (0,     0,   0)
COL_FPS      = (200, 200, 200)

THRESHOLD    = 0.68              # ArcFace cosine distance threshold
# ─────────────────────────────────────────────────────────────────────────────


class RecognitionWorker:
    """Runs DeepFace.find() in a background thread; never blocks the display."""

    def __init__(self, db_path: Path):
        self.db_path   = db_path
        self._lock     = threading.Lock()
        self._pending  = None          # frame queued for recognition
        self._busy     = False         # True while a recognition pass is running
        self._results  = []            # latest face list from last pass
        self._stop_evt = threading.Event()

        # Pre-build the face-embedding database (.pkl) on startup
        self._prebuild_db()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ── public API ────────────────────────────────────────────────────────────

    def submit(self, frame: np.ndarray):
        """Hand a new frame to the worker (non-blocking). Drops if busy."""
        with self._lock:
            if not self._busy:
                self._pending = frame.copy()

    def results(self) -> list:
        """Return the latest recognition results (thread-safe snapshot)."""
        with self._lock:
            return list(self._results)

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=5)

    # ── internals ─────────────────────────────────────────────────────────────

    def _prebuild_db(self):
        """Force DeepFace to build (or load) the .pkl embedding store."""
        print("⏳  Building face database embeddings …")
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            DeepFace.find(
                img_path   = dummy,
                db_path    = str(self.db_path),
                model_name = MODEL_NAME,
                detector_backend = DETECTOR_BACKEND,
                enforce_detection = False,
                silent     = True,
                return_type = "dict",
            )
        except Exception:
            pass
        print("✅  Database ready.")

    def _loop(self):
        while not self._stop_evt.is_set():
            frame = None
            with self._lock:
                if self._pending is not None:
                    frame = self._pending
                    self._pending = None
                    self._busy   = True

            if frame is None:
                time.sleep(0.005)
                continue

            faces = self._recognize(frame)

            with self._lock:
                self._results = faces
                self._busy    = False

    def _recognize(self, frame: np.ndarray) -> list:
        faces = []
        try:
            raw = DeepFace.find(
                img_path   = frame,
                db_path    = str(self.db_path),
                model_name = MODEL_NAME,
                detector_backend = DETECTOR_BACKEND,
                enforce_detection = False,
                silent     = True,
                return_type = "dict",
            )
            for face_matches in raw:
                if not face_matches:
                    continue
                best = face_matches[0]      # lowest distance = best match
                dist = float(best.get("distance", 1.0))
                name = (
                    Path(str(best["identity"])).stem
                    if dist <= THRESHOLD
                    else "Unknown"
                )
                faces.append({
                    "name"  : name,
                    "dist"  : dist,
                    "x"     : int(best.get("source_x", 0)),
                    "y"     : int(best.get("source_y", 0)),
                    "w"     : int(best.get("source_w", 0)),
                    "h"     : int(best.get("source_h", 0)),
                    "conf"  : float(best.get("confidence", 0.0)),
                })
        except Exception:
            pass
        return faces


def draw_overlay(frame: np.ndarray, faces: list, fps: float):
    """Render bounding boxes, labels and FPS counter onto frame."""
    h, w = frame.shape[:2]

    for f in faces:
        x, y, bw, bh = f["x"], f["y"], f["w"], f["h"]
        name, dist   = f["name"], f["dist"]
        matched      = name != "Unknown"
        colour       = COL_MATCH if matched else COL_UNKNOWN

        # ── bounding box ──────────────────────────────────────────────────
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), colour, 2)

        # ── corner accents (premium look) ─────────────────────────────────
        arm = max(12, bw // 6)
        thick = 3
        for (sx, sy, dx, dy) in [
            (x,      y,      1,  1),
            (x + bw, y,     -1,  1),
            (x,      y + bh, 1, -1),
            (x + bw, y + bh,-1, -1),
        ]:
            cv2.line(frame, (sx, sy), (sx + dx * arm, sy), colour, thick)
            cv2.line(frame, (sx, sy), (sx, sy + dy * arm), colour, thick)

        # ── label background ──────────────────────────────────────────────
        label    = f"{name}  {dist:.2f}"
        font     = cv2.FONT_HERSHEY_DUPLEX
        scale    = 0.55
        (tw, th), baseline = cv2.getTextSize(label, font, scale, 1)
        pad  = 5
        lx   = x
        ly   = max(y - th - pad * 2, 0)
        cv2.rectangle(frame, (lx, ly), (lx + tw + pad * 2, ly + th + pad * 2 + baseline), colour, -1)
        cv2.putText(frame, label, (lx + pad, ly + th + pad), font, scale, (0, 0, 0), 1, cv2.LINE_AA)

    # ── FPS counter ───────────────────────────────────────────────────────────
    fps_label = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_label, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_FPS, 1, cv2.LINE_AA)


def main():
    if not DB_PATH.is_dir():
        raise FileNotFoundError(f"Database folder not found: {DB_PATH}")

    images = [p for p in DB_PATH.iterdir()
              if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not images:
        raise RuntimeError(f"No reference images found in {DB_PATH}")

    print(f"📂  Reference images : {len(images)}")

    # ── Open camera ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep buffer tiny → minimal latency
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (source=0)")

    worker = RecognitionWorker(DB_PATH)

    print("🎥  Press  Q  to quit.")

    fps_timer   = time.perf_counter()
    fps         = 0.0
    frame_count = 0
    SUBMIT_EVERY = 3          # send every Nth frame to the recognizer

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Frame capture failed — retrying …")
                time.sleep(0.01)
                continue

            frame_count += 1

            # ── Submit a frame for async recognition ──────────────────────
            if frame_count % SUBMIT_EVERY == 0:
                worker.submit(frame)

            # ── Overlay latest results (never blocks) ─────────────────────
            draw_overlay(frame, worker.results(), fps)

            # ── Display (GUI optional; fall back to headless) ─────────────
            global _GUI_OK
            if _GUI_OK:
                try:
                    cv2.imshow(WINDOW_TITLE, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break
                    if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    _GUI_OK = False
                    print("⚠️  No GUI support in this OpenCV build — running headless (Ctrl+C to stop).")

            # ── FPS calculation ───────────────────────────────────────────
            now = time.perf_counter()
            fps = 1.0 / max(now - fps_timer, 1e-6)
            fps_timer = now

    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down …")
        worker.stop()
        cap.release()
        if _GUI_OK:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()
