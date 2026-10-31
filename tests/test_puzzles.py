"""Puzzle parsing.

The one thing that must be right is which position a puzzle refers to. The
stored FEN is the position before the opponent's mistake, so encoding it
directly would build a gate that scores the wrong board every time and still
look perfectly healthy. The mate-in-one case below is the test that catches
that: if the position were resolved wrongly, the solution would not be mate.

Rows here are real, taken from lichess_db_puzzle.csv.
"""

import chess
import pytest

from chessdt.encoding import MOVE_TO_INDEX
from chessdt.puzzles import parse_puzzle

MATE_IN_ONE = {
    "PuzzleId": "0000H",
    "FEN": "r4rk1/pp3ppp/8/2bqp2b/3N4/P2Bn2P/1PPN1P2/R1BQ1RK1 w - - 1 15",
    "Moves": "d1h5 d5g2",
    "Rating": "1554",
    "Themes": "mate mateIn1 middlegame oneMove",
}

LONG_LINE = {
    "PuzzleId": "00008",
    "FEN": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
    "Moves": "f2g3 e6e7 b2b1 b3c1 b1c1 h6c1",
    "Rating": "1811",
    "Themes": "crushing hangingPiece long middlegame",
}


def test_the_puzzle_position_is_after_the_opponents_move():
    puzzle = parse_puzzle(MATE_IN_ONE)
    assert puzzle is not None

    # The raw FEN has white to move; the solver is black, after d1h5.
    assert chess.Board(MATE_IN_ONE["FEN"]).turn == chess.WHITE
    assert puzzle.board.turn == chess.BLACK
    assert puzzle.board != chess.Board(MATE_IN_ONE["FEN"])


def test_the_solution_actually_solves_the_puzzle():
    """If the position were resolved wrongly, this would not be mate."""
    puzzle = parse_puzzle(MATE_IN_ONE)
    board = puzzle.board.copy()
    board.push_uci(puzzle.solution)
    assert board.is_checkmate()


def test_the_scored_move_is_the_second_not_the_first():
    puzzle = parse_puzzle(MATE_IN_ONE)
    assert puzzle.solution == "d5g2"
    assert puzzle.solution != MATE_IN_ONE["Moves"].split()[0]


def test_metadata_is_carried_through():
    puzzle = parse_puzzle(LONG_LINE)
    assert puzzle.puzzle_id == "00008"
    assert puzzle.rating == 1811
    assert "hangingPiece" in puzzle.themes
    assert puzzle.line_plies == 6


def test_the_solution_is_in_the_move_vocabulary():
    for row in (MATE_IN_ONE, LONG_LINE):
        puzzle = parse_puzzle(row)
        assert puzzle.solution in MOVE_TO_INDEX
        assert puzzle.solution_index == MOVE_TO_INDEX[puzzle.solution]


def test_a_solver_of_either_colour_is_handled():
    assert parse_puzzle(MATE_IN_ONE).board.turn == chess.BLACK
    assert parse_puzzle(LONG_LINE).board.turn == chess.WHITE


# ── Rejections ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reason", "overrides"),
    [
        ("no moves at all", {"Moves": ""}),
        ("only the setup move", {"Moves": "d1h5"}),
        ("unparseable FEN", {"FEN": "not a position"}),
        ("setup move illegal here", {"Moves": "a1a8 d5g2"}),
        ("solution illegal after setup", {"Moves": "d1h5 a1a2"}),
        ("non-numeric rating", {"Rating": "high"}),
        ("malformed uci", {"Moves": "zz99 d5g2"}),
    ],
)
def test_malformed_rows_are_rejected_rather_than_guessed(reason, overrides):
    assert parse_puzzle({**MATE_IN_ONE, **overrides}) is None, reason


def test_a_missing_column_is_rejected():
    row = {k: v for k, v in MATE_IN_ONE.items() if k != "Rating"}
    assert parse_puzzle(row) is None
