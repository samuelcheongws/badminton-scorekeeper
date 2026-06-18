from dataclasses import dataclass
from typing import Literal

Side = Literal["left", "right"]
ServiceCourt = Literal["left", "right"]
POINTS_TO_WIN = 21
MAX_POINTS = 30
GAMES_TO_WIN = 2


@dataclass
class ScoreState:
    points: list[int]
    games: list[int]
    serving_side: Side
    service_court: ServiceCourt
    game_number: int
    match_complete: bool
    winner: Side | None


class ScoreEngine:
    def __init__(self, first_server: Side = "left"):
        self._first_server = first_server
        self._points = [0, 0]
        self._games = [0, 0]
        self._serving_side: Side = first_server
        self._game_number = 1
        self._match_complete = False
        self._winner: Side | None = None

    @property
    def state(self) -> ScoreState:
        idx = 0 if self._serving_side == "left" else 1
        server_pts = self._points[idx]
        court: ServiceCourt = "right" if server_pts % 2 == 0 else "left"
        return ScoreState(
            points=list(self._points),
            games=list(self._games),
            serving_side=self._serving_side,
            service_court=court,
            game_number=self._game_number,
            match_complete=self._match_complete,
            winner=self._winner,
        )

    def add_point(self, side: Side) -> ScoreState:
        if self._match_complete:
            return self.state
        idx = 0 if side == "left" else 1
        self._points[idx] += 1
        self._serving_side = side
        self._check_game_over()
        return self.state

    def _check_game_over(self) -> None:
        l, r = self._points
        winner_idx = None
        if l >= POINTS_TO_WIN and l - r >= 2:
            winner_idx = 0
        elif r >= POINTS_TO_WIN and r - l >= 2:
            winner_idx = 1
        elif l == MAX_POINTS:
            winner_idx = 0
        elif r == MAX_POINTS:
            winner_idx = 1
        if winner_idx is not None:
            self._games[winner_idx] += 1
            self._points = [0, 0]
            if self._games[winner_idx] >= GAMES_TO_WIN:
                self._match_complete = True
                self._winner = "left" if winner_idx == 0 else "right"
            else:
                self._game_number += 1

    def reset_game(self) -> ScoreState:
        if self._match_complete:
            return self.state
        self._points = [0, 0]
        return self.state

    def reset_match(self) -> ScoreState:
        self._points = [0, 0]
        self._games = [0, 0]
        self._serving_side = self._first_server
        self._game_number = 1
        self._match_complete = False
        self._winner = None
        return self.state
