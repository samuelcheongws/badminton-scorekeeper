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
            cv2.putText(frame, msg, (mx, my), FONT, 1.1, CYAN, 2, cv2.LINE_AA)

        return frame
