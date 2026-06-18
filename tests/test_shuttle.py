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
