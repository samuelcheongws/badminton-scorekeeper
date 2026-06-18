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
