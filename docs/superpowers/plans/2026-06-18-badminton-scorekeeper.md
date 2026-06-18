# Badminton Scorekeeper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-time badminton scorekeeper that tracks shuttle position via TrackNetV2, detects scoring events automatically, and renders a bottom-bar HUD overlay on the live video stream.

**Architecture:** iPhone 12 via Continuity Camera feeds frames into a TrackNetV2 shuttle tracker. Detected shuttle positions (in normalised court coordinates) flow into an EventDetector that identifies scoring events (landing, out-of-bounds). A ScoreEngine applies BWF rally point rules and feeds state to a HUD renderer that draws a bottom bar directly on the video frame.

**Tech Stack:** Python 3.11+, OpenCV (`opencv-python`), PyTorch 2.1+ (MPS backend on Apple Silicon), NumPy.

---

## File Map

| File | Responsibility |
|---|---|
| `requirements.txt` | Python dependencies |
| `engine/score.py` | BWF rules state machine — `ScoreEngine`, `ScoreState` |
| `engine/events.py` | Trajectory analysis — `EventDetector`, `Detection`, `ScoringEvent` |
| `tracker/court.py` | Court calibration + homography — `CourtCalibrator` |
| `tracker/tracknet_model.py` | TrackNetV2 PyTorch model definition |
| `tracker/shuttle.py` | TrackNetV2 inference wrapper — `ShuttleTracker`, `ShuttleDetection` |
| `ui/hud.py` | OpenCV overlay drawing — `HUDRenderer` |
| `main.py` | Entry point, main loop, keyboard/mouse handling |
| `tests/test_score.py` | Unit tests for ScoreEngine |
| `tests/test_events.py` | Unit tests for EventDetector |
| `tests/test_court.py` | Unit tests for CourtCalibrator |
| `tests/test_tracknet_model.py` | Shape test for TrackNetV2 |
| `tests/test_shuttle.py` | Unit tests for ShuttleTracker (mock model) |
| `tests/test_hud.py` | Smoke tests for HUDRenderer |

---

## Coordinate conventions (read before implementing)

- **`ScoringEvent.side`** = the side that **scored** (won the point).
- `ScoreEngine.add_point(side)` adds 1 point to `side`.
- Court normalised coords: `x=0` left baseline (P1 side), `x=1` right baseline (P2 side), `x≈0.5` net. `y=0` near sideline, `y=1` far sideline.
- Shuttle lands in P2's half (`x ≥ 0.5`): P2 failed to return → **P1 scored** → `ScoringEvent(side="left")`.
- Shuttle lands in P1's half (`x < 0.5`): P1 failed → **P2 scored** → `ScoringEvent(side="right")`.
- Shuttle exits past right baseline (`x > 1.0`): P1 hit too far → **P2 scored** → `ScoringEvent(side="right")`.
- Shuttle exits past left baseline (`x < 0.0`): P2 hit too far → **P1 scored** → `ScoringEvent(side="left")`.
- For sideline outs (`y < 0` or `y > 1`): use last `dx` direction — if moving right, P1 hit it → P2 scored → `side="right"`.

---

## Task 1: Project scaffold

**Files:**
- Create: `requirements.txt`
- Create: `engine/__init__.py`
- Create: `tracker/__init__.py`
- Create: `ui/__init__.py`
- Create: `tests/__init__.py`
- Create: `models/.gitkeep`

- [ ] **Step 1: Create directory structure and empty init files**

```bash
mkdir -p engine tracker ui tests models
touch engine/__init__.py tracker/__init__.py ui/__init__.py tests/__init__.py models/.gitkeep
```

- [ ] **Step 2: Write `requirements.txt`**

```
opencv-python>=4.8.0
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
pytest>=7.4.0
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 4: Verify Python version**

```bash
python --version
```

Expected: `Python 3.11.x` or higher.

- [ ] **Step 5: Add `.gitignore`**

```
__pycache__/
*.pyc
*.pth
.superpowers/
models/*.pth
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt engine/__init__.py tracker/__init__.py ui/__init__.py tests/__init__.py models/.gitkeep .gitignore
git commit -m "chore: project scaffold and dependencies"
```

---

## Task 2: ScoreEngine

**Files:**
- Create: `tests/test_score.py`
- Create: `engine/score.py`

### Types

```python
Side = Literal["left", "right"]
ServiceCourt = Literal["left", "right"]

@dataclass
class ScoreState:
    points: list[int]         # [left_points, right_points]
    games: list[int]          # [left_games, right_games]
    serving_side: Side
    service_court: ServiceCourt
    game_number: int
    match_complete: bool
    winner: Side | None
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_score.py
import pytest
from engine.score import ScoreEngine

def test_initial_state():
    e = ScoreEngine(first_server="left")
    s = e.state
    assert s.points == [0, 0]
    assert s.games == [0, 0]
    assert s.serving_side == "left"
    assert s.game_number == 1
    assert not s.match_complete
    assert s.winner is None

def test_add_point_increments_scorer():
    e = ScoreEngine(first_server="left")
    s = e.add_point("left")
    assert s.points == [1, 0]

def test_server_switches_to_winner():
    e = ScoreEngine(first_server="left")
    s = e.add_point("right")
    assert s.serving_side == "right"

def test_server_stays_when_server_wins():
    e = ScoreEngine(first_server="left")
    s = e.add_point("left")
    assert s.serving_side == "left"

def test_service_court_even_score_is_right():
    e = ScoreEngine(first_server="left")
    assert e.state.service_court == "right"  # score 0, even

def test_service_court_odd_score_is_left():
    e = ScoreEngine(first_server="left")
    e.add_point("left")  # server score = 1, odd
    assert e.state.service_court == "left"

def test_game_won_at_21():
    e = ScoreEngine(first_server="left")
    for _ in range(21):
        e.add_point("left")
    s = e.state
    assert s.games == [1, 0]
    assert s.points == [0, 0]
    assert s.game_number == 2

def test_deuce_requires_2_point_lead():
    e = ScoreEngine(first_server="left")
    for _ in range(20):
        e.add_point("left")
    for _ in range(20):
        e.add_point("right")
    e.add_point("left")  # 21–20, not a win
    assert e.state.points == [21, 20]
    e.add_point("left")  # 22–20, win
    assert e.state.games == [1, 0]

def test_cap_at_30_29():
    e = ScoreEngine(first_server="left")
    for _ in range(29):
        e.add_point("left")
    for _ in range(29):
        e.add_point("right")
    s = e.add_point("left")  # 30–29, game won by cap
    assert s.games == [1, 0]

def test_match_complete_after_2_games():
    e = ScoreEngine(first_server="left")
    for _ in range(21):
        e.add_point("left")
    for _ in range(21):
        e.add_point("left")
    s = e.state
    assert s.match_complete
    assert s.winner == "left"

def test_add_point_no_op_when_match_complete():
    e = ScoreEngine(first_server="left")
    for _ in range(21):
        e.add_point("left")
    for _ in range(21):
        e.add_point("left")
    s = e.add_point("right")
    assert s.match_complete
    assert s.games == [2, 0]

def test_reset_game_clears_points_only():
    e = ScoreEngine(first_server="left")
    e.add_point("left")
    e.add_point("left")
    s = e.reset_game()
    assert s.points == [0, 0]
    assert s.games == [0, 0]
    assert s.game_number == 1

def test_reset_game_no_op_when_match_complete():
    e = ScoreEngine(first_server="left")
    for _ in range(21):
        e.add_point("left")
    for _ in range(21):
        e.add_point("left")
    s = e.reset_game()
    assert s.match_complete  # still complete

def test_reset_match_clears_everything():
    e = ScoreEngine(first_server="left")
    for _ in range(21):
        e.add_point("left")
    e.add_point("right")
    s = e.reset_match()
    assert s.points == [0, 0]
    assert s.games == [0, 0]
    assert s.game_number == 1
    assert not s.match_complete
    assert s.winner is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_score.py -v
```

Expected: `ModuleNotFoundError: No module named 'engine.score'`

- [ ] **Step 3: Implement `engine/score.py`**

```python
from dataclasses import dataclass
from typing import Literal

Side = Literal["left", "right"]
ServiceCourt = Literal["left", "right"]
POINTS_TO_WIN = 21
MAX_POINTS = 30
GAMES_TO_WIN = 2

@dataclass
class ScoreState:
    points: list[int]
    games: list[int]
    serving_side: Side
    service_court: ServiceCourt
    game_number: int
    match_complete: bool
    winner: Side | None

class ScoreEngine:
    def __init__(self, first_server: Side = "left"):
        self._first_server = first_server
        self._points = [0, 0]
        self._games = [0, 0]
        self._serving_side: Side = first_server
        self._game_number = 1
        self._match_complete = False
        self._winner: Side | None = None

    @property
    def state(self) -> ScoreState:
        idx = 0 if self._serving_side == "left" else 1
        server_pts = self._points[idx]
        court: ServiceCourt = "right" if server_pts % 2 == 0 else "left"
        return ScoreState(
            points=list(self._points),
            games=list(self._games),
            serving_side=self._serving_side,
            service_court=court,
            game_number=self._game_number,
            match_complete=self._match_complete,
            winner=self._winner,
        )

    def add_point(self, side: Side) -> ScoreState:
        if self._match_complete:
            return self.state
        idx = 0 if side == "left" else 1
        self._points[idx] += 1
        self._serving_side = side
        self._check_game_over()
        return self.state

    def _check_game_over(self) -> None:
        l, r = self._points
        winner_idx = None
        if l >= POINTS_TO_WIN and l - r >= 2:
            winner_idx = 0
        elif r >= POINTS_TO_WIN and r - l >= 2:
            winner_idx = 1
        elif l == MAX_POINTS:
            winner_idx = 0
        elif r == MAX_POINTS:
            winner_idx = 1
        if winner_idx is not None:
            self._games[winner_idx] += 1
            self._points = [0, 0]
            self._game_number += 1
            if self._games[winner_idx] >= GAMES_TO_WIN:
                self._match_complete = True
                self._winner = "left" if winner_idx == 0 else "right"

    def reset_game(self) -> ScoreState:
        if self._match_complete:
            return self.state
        self._points = [0, 0]
        return self.state

    def reset_match(self) -> ScoreState:
        self._points = [0, 0]
        self._games = [0, 0]
        self._serving_side = self._first_server
        self._game_number = 1
        self._match_complete = False
        self._winner = None
        return self.state
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_score.py -v
```

Expected: 14 tests pass.

- [ ] **Step 5: Commit**

```bash
git add engine/score.py tests/test_score.py
git commit -m "feat: BWF rally point ScoreEngine with full deuce and match rules"
```

---

## Task 3: EventDetector

**Files:**
- Create: `tests/test_events.py`
- Create: `engine/events.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_events.py
import time
import pytest
from engine.events import EventDetector, Detection, ScoringEvent

def det(x, y, conf=0.9, t=0.0):
    return Detection(x=x, y=y, confidence=conf, timestamp=t)

def test_no_event_on_none_with_empty_buffer():
    d = EventDetector()
    assert d.update(None) is None

def test_no_event_while_tracking():
    d = EventDetector(conf_threshold=0.5, min_tracked_frames=3)
    for i in range(3):
        assert d.update(det(0.75, 0.5, t=float(i))) is None

def test_landing_in_p2_half_scores_left():
    d = EventDetector(conf_threshold=0.5, min_tracked_frames=3)
    for i in range(3):
        d.update(det(0.75, 0.5, t=float(i) * 0.033))
    event = d.update(None)
    assert event is not None
    assert event.side == "left"
    assert event.event_type == "landing"

def test_landing_in_p1_half_scores_right():
    d = EventDetector(conf_threshold=0.5, min_tracked_frames=3)
    for i in range(3):
        d.update(det(0.25, 0.5, t=float(i) * 0.033))
    event = d.update(None)
    assert event is not None
    assert event.side == "right"

def test_out_past_right_baseline_scores_right():
    d = EventDetector(conf_threshold=0.5, min_tracked_frames=1)
    d.update(det(0.5, 0.5, t=0.0))
    event = d.update(det(1.2, 0.5, t=0.1))
    assert event is not None
    assert event.side == "right"
    assert event.event_type == "out"

def test_out_past_left_baseline_scores_left():
    d = EventDetector(conf_threshold=0.5, min_tracked_frames=1)
    d.update(det(0.5, 0.5, t=0.0))
    event = d.update(det(-0.1, 0.5, t=0.1))
    assert event is not None
    assert event.side == "left"

def test_debounce_suppresses_second_event():
    d = EventDetector(conf_threshold=0.5, min_tracked_frames=3, debounce_s=1.5)
    for i in range(3):
        d.update(det(0.75, 0.5, t=float(i) * 0.033))
    e1 = d.update(None)
    assert e1 is not None
    # second sequence immediately after
    for i in range(3):
        d.update(det(0.25, 0.5, t=0.2 + float(i) * 0.033))
    e2 = d.update(None)
    assert e2 is None

def test_low_confidence_detections_ignored():
    d = EventDetector(conf_threshold=0.5, min_tracked_frames=3)
    for i in range(5):
        d.update(det(0.75, 0.5, conf=0.3, t=float(i) * 0.033))
    event = d.update(None)
    assert event is None

def test_reset_clears_state():
    d = EventDetector(conf_threshold=0.5, min_tracked_frames=3)
    for i in range(3):
        d.update(det(0.75, 0.5, t=float(i) * 0.033))
    d.reset()
    event = d.update(None)
    assert event is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_events.py -v
```

Expected: `ModuleNotFoundError: No module named 'engine.events'`

- [ ] **Step 3: Implement `engine/events.py`**

```python
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

@dataclass
class Detection:
    x: float          # normalised court coord [0,1], may be outside if out-of-bounds
    y: float
    confidence: float
    timestamp: float

@dataclass
class ScoringEvent:
    side: Literal["left", "right"]        # side that SCORED
    event_type: Literal["landing", "out"]

class EventDetector:
    def __init__(
        self,
        debounce_s: float = 1.5,
        conf_threshold: float = 0.5,
        window_size: int = 15,
        min_tracked_frames: int = 3,
    ):
        self._debounce_s = debounce_s
        self._conf_threshold = conf_threshold
        self._min_tracked_frames = min_tracked_frames
        self._buf: deque[Detection] = deque(maxlen=window_size)
        self._last_event_t: float = -999.0

    def update(self, detection: Detection | None) -> ScoringEvent | None:
        now = time.monotonic()

        if detection is not None and detection.confidence >= self._conf_threshold:
            # Check out-of-bounds immediately
            if not (0.0 <= detection.x <= 1.0 and 0.0 <= detection.y <= 1.0):
                side = self._out_side(detection)
                return self._emit(side, "out", now)
            self._buf.append(detection)
            return None

        # Shuttle disappeared — check for landing
        high = [d for d in self._buf if d.confidence >= self._conf_threshold]
        if len(high) >= self._min_tracked_frames:
            last = high[-1]
            if 0.0 <= last.x <= 1.0 and 0.0 <= last.y <= 1.0:
                side = "left" if last.x >= 0.5 else "right"
                event = self._emit(side, "landing", now)
                self._buf.clear()
                return event
        self._buf.clear()
        return None

    def _out_side(self, d: Detection) -> Literal["left", "right"]:
        # Past right baseline: P1 hit it out → P2 scores (right)
        if d.x > 1.0:
            return "right"
        # Past left baseline: P2 hit it out → P1 scores (left)
        if d.x < 0.0:
            return "left"
        # Sideline out: use direction of travel from buffer
        if len(self._buf) >= 1:
            dx = d.x - self._buf[-1].x
            return "right" if dx > 0 else "left"
        return "right"

    def _emit(
        self,
        side: Literal["left", "right"],
        event_type: Literal["landing", "out"],
        now: float,
    ) -> ScoringEvent | None:
        if now - self._last_event_t < self._debounce_s:
            return None
        self._last_event_t = now
        self._buf.clear()
        return ScoringEvent(side=side, event_type=event_type)

    def reset(self) -> None:
        self._buf.clear()
        self._last_event_t = -999.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_events.py -v
```

Expected: 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add engine/events.py tests/test_events.py
git commit -m "feat: EventDetector with landing, out-of-bounds and debounce logic"
```

---

## Task 4: CourtCalibrator

**Files:**
- Create: `tests/test_court.py`
- Create: `tracker/court.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_court.py
import pytest
from tracker.court import CourtCalibrator

CORNERS = [(100, 50), (500, 50), (500, 400), (100, 400)]  # TL, TR, BR, BL

def calibrated():
    c = CourtCalibrator("singles")
    for x, y in CORNERS:
        c.on_click(x, y)
    return c

def test_not_calibrated_initially():
    assert not CourtCalibrator("singles").is_calibrated

def test_calibrated_after_4_clicks():
    c = CourtCalibrator("singles")
    results = [c.on_click(x, y) for x, y in CORNERS]
    assert results[-1] is True
    assert c.is_calibrated

def test_on_click_returns_false_before_4th():
    c = CourtCalibrator("singles")
    assert c.on_click(100, 50) is False
    assert c.on_click(500, 50) is False
    assert c.on_click(500, 400) is False

def test_to_court_coords_top_left_near_origin():
    c = calibrated()
    cx, cy = c.to_court_coords(100, 50)
    assert abs(cx) < 0.05
    assert abs(cy) < 0.05

def test_to_court_coords_bottom_right_near_one():
    c = calibrated()
    cx, cy = c.to_court_coords(500, 400)
    assert abs(cx - 1.0) < 0.05
    assert abs(cy - 1.0) < 0.05

def test_is_in_court_centre():
    c = calibrated()
    assert c.is_in_court(0.5, 0.5)

def test_is_out_of_court_right():
    c = calibrated()
    assert not c.is_in_court(1.5, 0.5)

def test_is_out_of_court_left():
    c = calibrated()
    assert not c.is_in_court(-0.1, 0.5)

def test_get_corners_returns_clicks():
    c = CourtCalibrator("singles")
    c.on_click(100, 50)
    c.on_click(500, 50)
    assert c.get_corners() == [(100, 50), (500, 50)]

def test_raises_before_calibrated():
    c = CourtCalibrator("singles")
    with pytest.raises(RuntimeError):
        c.to_court_coords(200, 200)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_court.py -v
```

Expected: `ModuleNotFoundError: No module named 'tracker.court'`

- [ ] **Step 3: Implement `tracker/court.py`**

```python
from typing import Literal
import numpy as np
import cv2

MatchType = Literal["singles", "doubles"]

class CourtCalibrator:
    def __init__(self, match_type: MatchType):
        self._match_type = match_type
        self._corners: list[tuple[int, int]] = []
        self._homography: np.ndarray | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._homography is not None

    def on_click(self, x: int, y: int) -> bool:
        if len(self._corners) >= 4:
            return True
        self._corners.append((x, y))
        if len(self._corners) == 4:
            self._compute()
            return True
        return False

    def _compute(self) -> None:
        src = np.array(self._corners, dtype=np.float32)
        dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        self._homography, _ = cv2.findHomography(src, dst)

    def to_court_coords(self, px: int, py: int) -> tuple[float, float]:
        if self._homography is None:
            raise RuntimeError("CourtCalibrator is not calibrated yet")
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._homography)
        return float(result[0, 0, 0]), float(result[0, 0, 1])

    def is_in_court(self, cx: float, cy: float) -> bool:
        return 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0

    def get_corners(self) -> list[tuple[int, int]]:
        return list(self._corners)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_court.py -v
```

Expected: 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tracker/court.py tests/test_court.py
git commit -m "feat: CourtCalibrator with 4-corner homography mapping"
```

---

## Task 5: TrackNetV2 model definition

**Files:**
- Create: `tests/test_tracknet_model.py`
- Create: `tracker/tracknet_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tracknet_model.py
import torch
from tracker.tracknet_model import TrackNetV2, INPUT_W, INPUT_H

def test_output_shape():
    model = TrackNetV2()
    x = torch.zeros(1, 9, INPUT_H, INPUT_W)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, INPUT_H, INPUT_W)

def test_output_in_0_1():
    model = TrackNetV2()
    x = torch.rand(1, 9, INPUT_H, INPUT_W)
    with torch.no_grad():
        out = model(x)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tracknet_model.py -v
```

Expected: `ModuleNotFoundError: No module named 'tracker.tracknet_model'`

- [ ] **Step 3: Implement `tracker/tracknet_model.py`**

```python
import torch
import torch.nn as nn

INPUT_W = 512
INPUT_H = 288

class _Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)

class TrackNetV2(nn.Module):
    """
    Input:  (B, 9, H, W)  — 3 RGB frames stacked channel-wise
    Output: (B, 1, H, W)  — shuttle confidence heatmap in [0, 1]
    """

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(_Block(9, 64), _Block(64, 64))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(_Block(64, 128), _Block(128, 128))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(_Block(128, 256), _Block(256, 256), _Block(256, 256))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = nn.Sequential(_Block(256, 512), _Block(512, 512), _Block(512, 512))
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(_Block(768, 256), _Block(256, 256), _Block(256, 256))
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(_Block(384, 128), _Block(128, 128))
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = nn.Sequential(_Block(192, 64), _Block(64, 64))
        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        d1 = self.dec1(torch.cat([self.up1(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e2], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d3))
```

Skip connections channel counts: enc4(512) + enc3(256) = 768 → dec1; dec1(256) + enc2(128) = 384 → dec2; dec2(128) + enc1(64) = 192 → dec3.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tracknet_model.py -v
```

Expected: 2 tests pass. This will take 10–20s as PyTorch compiles the model.

- [ ] **Step 5: Download pretrained weights**

The official weights are trained on professional badminton video and available from the TrackNetV2 repository at: https://github.com/yastrebksv/TrackNetV2

Download the `.pth` file and place it at `models/tracknet_weights.pth`. The weights must match the `TrackNetV2` architecture defined above.

- [ ] **Step 6: Commit**

```bash
git add tracker/tracknet_model.py tests/test_tracknet_model.py
git commit -m "feat: TrackNetV2 model definition (VGG encoder + skip-connection decoder)"
```

---

## Task 6: ShuttleTracker

**Files:**
- Create: `tests/test_shuttle.py`
- Create: `tracker/shuttle.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_shuttle.py
import numpy as np
import torch
import pytest
from unittest.mock import MagicMock
from tracker.shuttle import ShuttleTracker
from tracker.tracknet_model import INPUT_W, INPUT_H

def blank_frame(h=1080, w=1920):
    return np.zeros((h, w, 3), dtype=np.uint8)

def mock_tracker(heatmap: torch.Tensor, conf_threshold: float = 0.5) -> ShuttleTracker:
    mock = MagicMock(return_value=heatmap)
    return ShuttleTracker(weights_path="", conf_threshold=conf_threshold, _model=mock)

def test_returns_none_with_fewer_than_3_frames():
    t = mock_tracker(torch.zeros(1, 1, INPUT_H, INPUT_W))
    assert t.update(blank_frame()) is None
    assert t.update(blank_frame()) is None

def test_returns_none_when_peak_below_threshold():
    heatmap = torch.zeros(1, 1, INPUT_H, INPUT_W)
    heatmap[0, 0, 144, 256] = 0.3
    t = mock_tracker(heatmap, conf_threshold=0.5)
    t.update(blank_frame())
    t.update(blank_frame())
    result = t.update(blank_frame())
    assert result is None

def test_returns_detection_when_peak_above_threshold():
    heatmap = torch.zeros(1, 1, INPUT_H, INPUT_W)
    heatmap[0, 0, 144, 256] = 0.9
    t = mock_tracker(heatmap, conf_threshold=0.5)
    t.update(blank_frame())
    t.update(blank_frame())
    result = t.update(blank_frame())
    assert result is not None
    assert result.confidence == pytest.approx(0.9, abs=0.01)

def test_detection_coords_scaled_to_frame():
    heatmap = torch.zeros(1, 1, INPUT_H, INPUT_W)
    heatmap[0, 0, 0, 0] = 0.9  # top-left of heatmap
    t = mock_tracker(heatmap)
    t.update(blank_frame())
    t.update(blank_frame())
    result = t.update(blank_frame())
    assert result is not None
    assert result.x == 0
    assert result.y == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_shuttle.py -v
```

Expected: `ModuleNotFoundError: No module named 'tracker.shuttle'`

- [ ] **Step 3: Implement `tracker/shuttle.py`**

```python
import cv2
import numpy as np
import torch
from dataclasses import dataclass
from tracker.tracknet_model import TrackNetV2, INPUT_W, INPUT_H

@dataclass
class ShuttleDetection:
    x: int            # pixel x in original frame
    y: int            # pixel y in original frame
    confidence: float

class ShuttleTracker:
    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = 0.5,
        device: str | None = None,
        _model=None,
    ):
        if _model is not None:
            self._model = _model
            self._device = torch.device("cpu")
        else:
            if device is None:
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._device = torch.device(device)
            model = TrackNetV2().to(self._device)
            model.load_state_dict(torch.load(weights_path, map_location=self._device))
            model.eval()
            self._model = model
        self._conf_threshold = conf_threshold
        self._buffer: list[np.ndarray] = []

    def update(self, frame: np.ndarray) -> ShuttleDetection | None:
        """Add frame to rolling buffer; run inference when 3 frames are available."""
        self._buffer.append(frame)
        if len(self._buffer) > 3:
            self._buffer.pop(0)
        if len(self._buffer) < 3:
            return None

        h, w = frame.shape[:2]
        frames_rgb = [
            cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (INPUT_W, INPUT_H))
            for f in self._buffer
        ]
        stacked = np.concatenate([f.transpose(2, 0, 1) for f in frames_rgb], axis=0)
        tensor = torch.from_numpy(stacked).float().unsqueeze(0).div(255.0)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            heatmap = self._model(tensor)

        hm = heatmap.squeeze().cpu().numpy()
        conf = float(hm.max())
        if conf < self._conf_threshold:
            return None

        hy, hx = np.unravel_index(int(np.argmax(hm)), hm.shape)
        px = int(hx * w / INPUT_W)
        py = int(hy * h / INPUT_H)
        return ShuttleDetection(x=px, y=py, confidence=conf)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_shuttle.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tracker/shuttle.py tests/test_shuttle.py
git commit -m "feat: ShuttleTracker wrapping TrackNetV2 with MPS/CPU fallback"
```

---

## Task 7: HUD Renderer

**Files:**
- Create: `tests/test_hud.py`
- Create: `ui/hud.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hud.py
import numpy as np
from ui.hud import HUDRenderer
from engine.score import ScoreEngine

def blank(h=1080, w=1920):
    return np.zeros((h, w, 3), dtype=np.uint8)

def test_draw_returns_same_shape():
    r = HUDRenderer()
    frame = blank()
    result = r.draw(frame.copy(), ScoreEngine("left").state)
    assert result.shape == frame.shape

def test_draw_modifies_frame():
    r = HUDRenderer()
    frame = blank()
    result = r.draw(frame.copy(), ScoreEngine("left").state)
    assert not np.array_equal(result, frame)

def test_draw_match_complete_does_not_raise():
    r = HUDRenderer()
    e = ScoreEngine("left")
    for _ in range(21):
        e.add_point("left")
    for _ in range(21):
        e.add_point("left")
    result = r.draw(blank(), e.state)
    assert result.shape == (1080, 1920, 3)

def test_draw_deuce_score_does_not_raise():
    r = HUDRenderer()
    e = ScoreEngine("left")
    for _ in range(20):
        e.add_point("left")
    for _ in range(20):
        e.add_point("right")
    result = r.draw(blank(), e.state)
    assert result.shape == (1080, 1920, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_hud.py -v
```

Expected: `ModuleNotFoundError: No module named 'ui.hud'`

- [ ] **Step 3: Implement `ui/hud.py`**

```python
import cv2
import numpy as np
from engine.score import ScoreState

BAR_H = 80
ALPHA = 0.82
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREY = (160, 160, 160)
DARK = (70, 70, 70)
CYAN = (255, 229, 0)
BLUE = (255, 158, 77)    # BGR
RED = (107, 107, 255)    # BGR

class HUDRenderer:
    def draw(self, frame: np.ndarray, state: ScoreState) -> np.ndarray:
        h, w = frame.shape[:2]
        y0 = h - BAR_H
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, frame)

        top = y0 + 22
        bot = y0 + 62

        # — Player 1 (left) —
        p1_label = ("● " if state.serving_side == "left" else "") + "PLAYER 1"
        cv2.putText(frame, p1_label, (20, top), FONT, 0.55, BLUE, 1, cv2.LINE_AA)
        cv2.putText(frame, str(state.points[0]), (20, bot), FONT, 1.4, WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"games {state.games[0]}", (80, bot), FONT, 0.45, GREY, 1, cv2.LINE_AA)

        # — Player 2 (right) —
        p2_label = "PLAYER 2" + (" ●" if state.serving_side == "right" else "")
        (tw, _), _ = cv2.getTextSize(p2_label, FONT, 0.55, 1)
        cv2.putText(frame, p2_label, (w - tw - 20, top), FONT, 0.55, RED, 1, cv2.LINE_AA)
        p2_score = str(state.points[1])
        (sw, _), _ = cv2.getTextSize(p2_score, FONT, 1.4, 2)
        cv2.putText(frame, p2_score, (w - sw - 100, bot), FONT, 1.4, WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"games {state.games[1]}", (w - 90, bot), FONT, 0.45, GREY, 1, cv2.LINE_AA)

        # — Centre —
        game_lbl = f"GAME {state.game_number}"
        (gw, _), _ = cv2.getTextSize(game_lbl, FONT, 0.5, 1)
        cv2.putText(frame, game_lbl, (w // 2 - gw // 2, top), FONT, 0.5, DARK, 1, cv2.LINE_AA)
        hint = "R = reset game   RR = reset match"
        (hw, _), _ = cv2.getTextSize(hint, FONT, 0.35, 1)
        cv2.putText(frame, hint, (w // 2 - hw // 2, top + 22), FONT, 0.35, DARK, 1, cv2.LINE_AA)

        # — Match complete banner —
        if state.match_complete:
            winner = "Player 1" if state.winner == "left" else "Player 2"
            msg = f"MATCH COMPLETE  {winner} wins"
            (mw, mh), _ = cv2.getTextSize(msg, FONT, 1.1, 2)
            mx, my = w // 2 - mw // 2, h // 2
            cv2.rectangle(frame, (mx - 20, my - mh - 10), (mx + mw + 20, my + 10), (0, 0, 0), -1)
            cv2.putText(frame, msg, (mx, my), FONT, 1.1, (0, 229, 255), 2, cv2.LINE_AA)

        return frame
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_hud.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add ui/hud.py tests/test_hud.py
git commit -m "feat: HUDRenderer with bottom-bar overlay and match-complete banner"
```

---

## Task 8: Main app

**Files:**
- Create: `main.py`

No automated tests — this wires all components together and is verified manually with the camera.

- [ ] **Step 1: Run the full test suite before touching main.py**

```bash
pytest tests/ -v
```

Expected: all tests pass (29 total).

- [ ] **Step 2: Implement `main.py`**

```python
import sys
import time
import cv2
import numpy as np
from tracker.court import CourtCalibrator
from tracker.shuttle import ShuttleTracker
from engine.events import EventDetector, Detection
from engine.score import ScoreEngine
from ui.hud import HUDRenderer

WEIGHTS = "models/tracknet_weights.pth"
WINDOW = "Badminton Scorekeeper"

def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}. Check Continuity Camera is active.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap

def ask_match_type() -> str:
    while True:
        answer = input("Match type? [1] Singles  [2] Doubles: ").strip()
        if answer in ("1", "2"):
            return "singles" if answer == "1" else "doubles"

def main() -> None:
    match_type = ask_match_type()
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    cap = open_camera(cam_index)
    calibrator = CourtCalibrator(match_type)
    tracker = ShuttleTracker(WEIGHTS)
    detector = EventDetector()
    score = ScoreEngine(first_server="left")
    hud = HUDRenderer()

    state = {"calibrating": True}
    last_r = -999.0

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state["calibrating"]:
            if calibrator.on_click(x, y):
                state["calibrating"] = False

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if state["calibrating"]:
            display = frame.copy()
            n = len(calibrator.get_corners())
            cv2.putText(
                display,
                f"Click court corners ({n}/4):  top-left -> top-right -> bottom-right -> bottom-left",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 229, 255), 2, cv2.LINE_AA,
            )
            for cx, cy in calibrator.get_corners():
                cv2.circle(display, (cx, cy), 10, (0, 255, 0), -1)
            cv2.imshow(WINDOW, display)
        else:
            shuttle = tracker.update(frame)
            detection = None
            if shuttle is not None:
                cx, cy = calibrator.to_court_coords(shuttle.x, shuttle.y)
                detection = Detection(x=cx, y=cy, confidence=shuttle.confidence, timestamp=time.monotonic())

            event = detector.update(detection)
            if event is not None:
                score.add_point(event.side)

            cv2.imshow(WINDOW, hud.draw(frame.copy(), score.state))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            now = time.monotonic()
            if now - last_r < 1.0:
                score.reset_match()
                detector.reset()
            else:
                score.reset_game()
            last_r = now

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the full test suite to confirm nothing is broken**

```bash
pytest tests/ -v
```

Expected: all tests still pass.

- [ ] **Step 4: Smoke test with camera**

```bash
python main.py
```

1. Type `1` (singles) and press Enter.
2. OpenCV window opens with live feed.
3. Click the 4 court corners. Green dots appear at each click.
4. After 4th click, tracking begins and the bottom-bar HUD appears.
5. Verify score increments when a rally ends (shuttle lands).
6. Press `R` once — game score resets to 0–0.
7. Press `R` twice quickly — full match resets.
8. Press `Q` — window closes cleanly.

If the iPhone isn't camera index 0, try: `python main.py 1` (or 2, 3, etc.).

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: main app loop wiring all components together"
```

---

## Self-review checklist

- [x] Startup singles/doubles prompt → `ask_match_type()` in main.py
- [x] Court calibration → `CourtCalibrator.on_click`, mouse callback
- [x] TrackNetV2 shuttle tracking → Task 5 + Task 6
- [x] Normalised court coords + homography → `to_court_coords` + EventDetector
- [x] BWF scoring rules (21, deuce, cap, best of 3) → `ScoreEngine._check_game_over`
- [x] Serving side indicator → `ScoreState.serving_side`, `service_court`
- [x] Bottom-bar HUD with points, games, serving dot → `HUDRenderer`
- [x] Match complete banner → `HUDRenderer` match_complete branch
- [x] Reset game (`R`) and reset match (`RR`) → main.py keyboard handler + `ScoreEngine`
- [x] No-op reset when match is complete → `ScoreEngine.reset_game`
- [x] MPS backend on Apple Silicon → `ShuttleTracker.__init__`
- [x] Quit on Q / Esc → main.py key handler
