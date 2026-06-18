# Badminton Scorekeeper — Design Spec
_Date: 2026-06-18_

## Overview

A real-time badminton scorekeeper that processes a live video feed using computer vision to automatically detect scoring events and overlays the score on the video stream. Supports singles and doubles matches. Runs on a MacBook Air with an iPhone 12 as the camera source.

---

## Hardware Setup

**Camera:** iPhone 12 via Apple Continuity Camera (macOS Ventura 13+ / iOS 16+). Appears as a standard webcam in OpenCV (`cv2.VideoCapture()`). Connect via USB (Lightning) for reliability or wirelessly over Wi-Fi/Bluetooth.

**Placement:** Tripod on the sideline at mid-court, 1.8–2.5m high, 1x zoom (no wide-angle), slight downward tilt to capture both baselines. The full court fits in frame. No encroachment on adjacent courts.

**Processing:** MacBook Air (Apple Silicon). TrackNetV2 runs on the MPS (Metal Performance Shaders) backend for GPU-accelerated inference on Apple Silicon.

---

## Architecture

```
iPhone 12 (Continuity Camera)
        │ USB / WiFi
        ▼
  VideoCapture (OpenCV)
  cv2.VideoCapture() — 1080p / 30fps
  Maintains rolling buffer of 3 consecutive frames
        │
        ├──────────────────────────────────► CourtCalibrator (runs once at startup)
        │                                    User clicks 4 court corners
        │                                    Computes homography: pixel → court coords
        │                                    Outputs court mask for in/out detection
        │
        ▼
  ShuttleTracker (TrackNetV2 / PyTorch)
  Input: 3 frames → Output: xy confidence heatmap
  Pretrained weights (professional badminton footage)
  Runs on Apple Silicon MPS backend
        │
        ▼ xy coords + confidence
  EventDetector
  Analyses shuttle trajectory over a sliding window
  Detects: landing in court · shuttle out of bounds · net fault
        │
        ▼ scoring event + side
  ScoreEngine
  Applies BWF rally point rules
  Tracks: points, games won, serving side, deuce state
  Handles reset (keyboard 'R')
        │
        ▼ score state
  HUD Renderer (OpenCV)
  Draws bottom-bar overlay on raw frame:
    — Player names · current game points · games won · serving dot
  cv2.imshow() output window on MacBook screen
```

---

## Components

### VideoCapture
- `cv2.VideoCapture()` targeting the iPhone via Continuity Camera
- Maintains a rolling deque of the last 3 frames (required by TrackNetV2)
- Target: 1080p at 30fps

### CourtCalibrator (`tracker/court.py`)
- Runs once on first frame at startup
- Freezes the frame and prompts: "Click 4 court corners: top-left → top-right → bottom-right → bottom-left"
- Computes a homography matrix mapping pixel coordinates to normalised court coordinates (0.0–1.0 on each axis)
- Generates a binary court mask used to filter out detections outside the court boundary

### ShuttleTracker (`tracker/shuttle.py`)
- Wraps the pretrained TrackNetV2 PyTorch model
- Input: 3 consecutive RGB frames (resized to model input size)
- Output: 2D heatmap → peak coordinate (x, y) + confidence score
- Runs on MPS backend (Apple Silicon GPU); falls back to CPU if unavailable
- Pretrained weights stored at `models/tracknet_weights.pth`

### EventDetector (`engine/events.py`)
- Receives a time-series of (x, y, confidence) shuttle positions
- Uses a sliding window to detect trajectory events:
  - **Landing in court:** trajectory peaks, descends to floor level, position is inside court boundary → point to opponent
  - **Out of bounds:** shuttle position crosses sideline/baseline in normalised court coords → point to opponent
  - **Net fault:** shuttle stops at net height (y ≈ net level) with near-zero horizontal velocity → point to opponent
- Emits a `ScoringEvent(side: "left" | "right", type: "landing" | "out" | "net")` when triggered
- Debounce: 1.5s cooldown after each event to prevent double-counting

### ScoreEngine (`engine/score.py`)
- State machine applying BWF rally point scoring rules:
  - First to 21, must lead by 2
  - At 20–20, play to 2-point lead; capped at 30–29
  - Best of 3 games
  - Server: winner of last point; service side determined by server's current score parity (even = right court, odd = left court)
- Exposes: `add_point(side)`, `reset_game()`, `reset_match()`, `state → ScoreState`
- Reset: 'R' key resets current game; double-tap 'R' (within 1s) resets full match

### HUD Renderer (`ui/hud.py`)
- **Layout:** Bottom bar, full-width, semi-transparent black background (`rgba(0,0,0,0.82)`)
- **Left:** Player 1 name · current game score (large) · games won · serving dot (●) if serving
- **Centre:** Current game number · "Press R to reset" hint
- **Right:** Player 2 name · current game score (large) · games won · serving dot if serving
- Drawn with `cv2.rectangle`, `cv2.putText`, `cv2.circle` directly onto the raw frame

---

## Scoring Rules Reference (BWF Rally Point)

| Situation | Rule |
|---|---|
| Normal play | First to 21, win by 2 |
| At 20–20 (deuce) | Continue until one side leads by 2 |
| Maximum score | 30–29 (cap) |
| Games | Best of 3 |
| Serving | Winner of previous rally serves |
| Service court | Even score → right court; odd score → left court |

---

## Startup Flow

1. App launches with a terminal prompt: `Match type? [1] Singles  [2] Doubles` — user types 1 or 2 and presses Enter. This determines service court width used in the court mask.
2. OpenCV window opens showing live feed.
3. **Court calibration:** frame freezes, user clicks 4 corners (top-left → top-right → bottom-right → bottom-left). Green dots confirm each click. Homography computed on 4th click.
4. Live tracking begins. HUD appears at bottom of frame.
5. Scoring events are detected automatically and score updates in real time.
6. **Match completion:** when a player wins 2 games, the HUD displays "MATCH COMPLETE — [Player X] wins" in the centre of the frame. Scoring is frozen. Press 'R' twice to start a new match.

---

## Controls

| Key | Action |
|---|---|
| `R` | Reset current game score to 0–0 |
| `R R` (double-tap within 1s) | Reset full match |
| `Q` / `Esc` | Quit |

---

## Tech Stack

| Library | Purpose |
|---|---|
| Python 3.11+ | Runtime |
| `opencv-python` | Video capture, frame rendering, HUD drawing |
| `torch` + `torchvision` | TrackNetV2 inference (MPS backend on Apple Silicon) |
| `numpy` | Coordinate math, homography computation |
| `scipy` | Trajectory smoothing (Savitzky-Golay filter) |

No web server, no database, no frontend framework.

---

## Project Structure

```
badminton-scorekeeper/
├── main.py                              # entry point, main loop
├── tracker/
│   ├── shuttle.py                       # TrackNetV2 wrapper
│   └── court.py                         # calibration + homography
├── engine/
│   ├── events.py                        # trajectory → scoring event
│   └── score.py                         # BWF rules, state machine
├── ui/
│   └── hud.py                           # cv2 overlay drawing
├── models/
│   └── tracknet_weights.pth             # pretrained weights (downloaded at setup)
├── requirements.txt
└── README.md
```

---

## Known Limitations

- TrackNetV2 accuracy degrades in poor lighting or when the shuttle is occluded by players
- Serving side inference assumes standard BWF rules — does not handle non-standard serve orders after game restarts
- Court calibration must be redone if the camera is moved
- Singles vs. doubles distinction affects service court rules but not point detection; user sets match type at startup
