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

    try:
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
                else:
                    score.reset_game()
                detector.reset()
                last_r = now
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
