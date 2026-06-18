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
