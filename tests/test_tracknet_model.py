import torch
from tracker.tracknet_model import TrackNetV2, INPUT_W, INPUT_H

def test_output_shape():
    model = TrackNetV2().eval()
    x = torch.zeros(1, 9, INPUT_H, INPUT_W)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, INPUT_H, INPUT_W)

def test_output_in_0_1():
    torch.manual_seed(0)
    model = TrackNetV2().eval()
    x = torch.rand(1, 9, INPUT_H, INPUT_W)
    with torch.no_grad():
        out = model(x)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
