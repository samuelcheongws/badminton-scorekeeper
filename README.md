# Badminton Scorekeeper

Real-time badminton score tracking using computer vision. Point a camera at the court and the system automatically detects scoring events and overlays the score on the live video feed. Supports singles and doubles matches.

## How it works

1. iPhone 12 streams video to MacBook Air via Apple Continuity Camera
2. TrackNetV2 (a deep learning model built for badminton shuttle tracking) processes each frame
3. The shuttle trajectory is analysed to detect landing, out-of-bounds, and net fault events
4. Score updates automatically and is rendered as a HUD overlay on the video stream

## Hardware setup

- **MacBook Air** (Apple Silicon) — runs all inference and display
- **iPhone 12** — camera source via Continuity Camera (USB or Wi-Fi)
- **Tripod** — place iPhone on sideline at mid-court, 1.8–2.5m high, 1x zoom

Both devices must be signed into the same Apple ID with Wi-Fi and Bluetooth enabled. The iPhone appears automatically as a camera source in any app.

## Quick start

```bash
# Install dependencies
uv sync

# Download pretrained TrackNetV2 weights from the official repository:
# https://github.com/yastrebksv/TrackNetV2
# Place the downloaded .pth file at models/tracknet_weights.pth

# Run
uv run python main.py
```

On launch, you'll be prompted to select match type (singles or doubles). Then the first frame freezes. Click the 4 court corners (top-left → top-right → bottom-right → bottom-left) to calibrate the court boundaries. Live scoring begins immediately after.

## Controls

| Key | Action |
|---|---|
| `R` | Reset current game score |
| `R R` (double-tap) | Reset full match |
| `Q` / `Esc` | Quit |

## Scoring rules

Standard BWF rally point scoring:
- First to 21, win by 2 (deuce at 20–20, cap at 30–29)
- Best of 3 games
- Server is the winner of the previous rally
- Service court: even score → right court, odd score → left court

## Project structure

```
badminton-scorekeeper/
├── main.py          # entry point, main loop
├── tracker/
│   ├── shuttle.py   # TrackNetV2 wrapper
│   └── court.py     # court calibration + homography
├── engine/
│   ├── events.py    # shuttle trajectory → scoring event
│   └── score.py     # BWF rules state machine
├── ui/
│   └── hud.py       # score overlay rendering
├── models/
│   └── tracknet_weights.pth
└── requirements.txt
```

## Tech stack

- Python 3.11+
- OpenCV — video capture and HUD rendering
- PyTorch (MPS backend) — TrackNetV2 inference on Apple Silicon
- NumPy — coordinate math and homography
- SciPy — trajectory smoothing

## Limitations

- Accuracy degrades in poor lighting or when the shuttle is occluded
- Court calibration must be redone if the camera is moved
- Serving side inference follows standard BWF rules only
