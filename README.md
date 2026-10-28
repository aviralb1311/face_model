# Real-time Face Recognition

Zero-lag, flicker-free real-time face recognition using DeepFace (ArcFace + RetinaFace).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add reference images (`.jpg`/`.png`) to `face_database/`.

## Run

```bash
python main.py
```

Press `Q` or `Esc` to quit.

## Notes

- Requires a webcam.
- First run downloads model weights (~hundreds of MB).
- OpenCV GUI support is required for the preview window.
