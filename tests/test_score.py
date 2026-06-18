from engine.score import ScoreEngine

def test_initial_state():
    e = ScoreEngine(first_server="left")
    s = e.state
    assert s.points == [0, 0]
    assert s.games == [0, 0]
    assert s.serving_side == "left"
    assert s.game_number == 1
    assert not s.match_complete
    assert s.winner is None

def test_add_point_increments_scorer():
    e = ScoreEngine(first_server="left")
    s = e.add_point("left")
    assert s.points == [1, 0]

def test_server_switches_to_winner():
    e = ScoreEngine(first_server="left")
    s = e.add_point("right")
    assert s.serving_side == "right"

def test_server_stays_when_server_wins():
    e = ScoreEngine(first_server="left")
    s = e.add_point("left")
    assert s.serving_side == "left"

def test_service_court_even_score_is_right():
    e = ScoreEngine(first_server="left")
    assert e.state.service_court == "right"  # score 0, even

def test_service_court_odd_score_is_left():
    e = ScoreEngine(first_server="left")
    e.add_point("left")  # server score = 1, odd
    assert e.state.service_court == "left"

def test_game_won_at_21():
    e = ScoreEngine(first_server="left")
    for _ in range(21):
        e.add_point("left")
    s = e.state
    assert s.games == [1, 0]
    assert s.points == [0, 0]
    assert s.game_number == 2

def test_deuce_requires_2_point_lead():
    e = ScoreEngine(first_server="left")
    for _ in range(20):
        e.add_point("left")
    for _ in range(20):
        e.add_point("right")
    e.add_point("left")  # 21–20, not a win
    assert e.state.points == [21, 20]
    e.add_point("left")  # 22–20, win
    assert e.state.games == [1, 0]

def test_cap_at_30_29():
    # Reach 29-29 within a single game by alternating points after deuce (20-20)
    e = ScoreEngine(first_server="left")
    for _ in range(20):
        e.add_point("left")
    for _ in range(20):
        e.add_point("right")
    # Now at 20-20 (deuce); alternate to reach 29-29
    for _ in range(9):
        e.add_point("left")
        e.add_point("right")
    assert e.state.points == [29, 29]
    s = e.add_point("left")  # 30–29, game won by cap
    assert s.games == [1, 0]

def test_match_complete_after_2_games():
    e = ScoreEngine(first_server="left")
    for _ in range(21):
        e.add_point("left")
    for _ in range(21):
        e.add_point("left")
    s = e.state
    assert s.match_complete
    assert s.winner == "left"

def test_add_point_no_op_when_match_complete():
    e = ScoreEngine(first_server="left")
    for _ in range(21):
        e.add_point("left")
    for _ in range(21):
        e.add_point("left")
    s = e.add_point("right")
    assert s.match_complete
    assert s.games == [2, 0]

def test_reset_game_clears_points_only():
    e = ScoreEngine(first_server="left")
    e.add_point("left")
    e.add_point("left")
    s = e.reset_game()
    assert s.points == [0, 0]
    assert s.games == [0, 0]
    assert s.game_number == 1

def test_reset_game_no_op_when_match_complete():
    e = ScoreEngine(first_server="left")
    for _ in range(21):
        e.add_point("left")
    for _ in range(21):
        e.add_point("left")
    s = e.reset_game()
    assert s.match_complete  # still complete

def test_reset_match_clears_everything():
    e = ScoreEngine(first_server="right")  # use right so we can detect if it doesn't restore
    e.add_point("left")  # left scores, server switches to left
    assert e.state.serving_side == "left"  # confirm server switched
    s = e.reset_match()
    assert s.points == [0, 0]
    assert s.games == [0, 0]
    assert s.game_number == 1
    assert not s.match_complete
    assert s.winner is None
    assert s.serving_side == "right"  # restored to first_server

def test_initial_state_right_server():
    e = ScoreEngine(first_server="right")
    s = e.state
    assert s.serving_side == "right"
    assert s.service_court == "right"  # score 0, even

def test_game_number_stays_at_winning_game():
    e = ScoreEngine(first_server="left")
    for _ in range(21):
        e.add_point("left")   # game 1 won
    assert e.state.game_number == 2  # correctly advances to game 2
    for _ in range(21):
        e.add_point("left")   # game 2 won, match complete
    assert e.state.game_number == 2  # stays at 2, not 3
