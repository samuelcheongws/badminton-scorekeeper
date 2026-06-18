import cv2
import numpy as np
import torch
from collections import deque
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
            model.load_state_dict(torch.load(weights_path, map_location=self._device, weights_only=True))
            model.eval()
            self._model = model
        self._conf_threshold = conf_threshold
        self._buffer: deque[np.ndarray] = deque(maxlen=3)

    def update(self, frame: np.ndarray) -> ShuttleDetection | None:
        """Add frame to rolling buffer; run inference when 3 frames are available."""
        self._buffer.append(frame)
        if len(self._buffer) < 3:
            return None

        h, w = frame.shape[:2]
        frames_rgb = [
            cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (INPUT_W, INPUT_H))  # expects OpenCV BGR input
            for f in self._buffer
        ]
        stacked = np.concatenate([f.transpose(2, 0, 1) for f in frames_rgb], axis=0)
        tensor = torch.from_numpy(stacked).float().unsqueeze(0).div(255.0)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            heatmap = self._model(tensor)

        hm = heatmap[0, 0].cpu().numpy()
        conf = float(hm.max())
        if conf < self._conf_threshold:
            return None

        hy, hx = np.unravel_index(int(np.argmax(hm)), hm.shape)
        px = int(round(hx * (w - 1) / (INPUT_W - 1)))
        py = int(round(hy * (h - 1) / (INPUT_H - 1)))
        return ShuttleDetection(x=px, y=py, confidence=conf)
