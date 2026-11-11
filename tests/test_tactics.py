"""Mate-in-one detection."""

import chess

from chessdt.tactics import mating_moves


def test_scholars_mate_is_found():
    board = chess.Board()
    for san in ("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"):
        board.push_san(san)
    assert mating_moves(board) == {"h5f7"}


def test_a_quiet_position_has_no_mate():
    assert mating_moves(chess.Board()) == set()


def test_every_mating_move_is_returned_when_there_are_several():
    # Back rank: both the rook and the queen can mate on the eighth.
    board = chess.Board("6k1/5ppp/8/8/8/8/8/Q3R1K1 w - - 0 1")
    assert mating_moves(board) == {"a1a8", "e1e8"}


def test_a_check_that_can_be_answered_is_not_a_mate():
    """Ra8+ gives check, but the king simply steps away."""
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    assert board.gives_check(chess.Move.from_uci("a1a8"))
    assert mating_moves(board) == set()


def test_the_board_is_left_exactly_as_it_was():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/Q3R1K1 w - - 0 1")
    before = board.fen()
    mating_moves(board)
    assert board.fen() == before
