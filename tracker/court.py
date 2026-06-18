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
