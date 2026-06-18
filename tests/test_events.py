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
