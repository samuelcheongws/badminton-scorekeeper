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
