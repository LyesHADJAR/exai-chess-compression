"""Mate-in-one detection."""

import chess

from chessdt.encoding import MOVE_TO_INDEX, pack_board
from chessdt.tactics import (
    NO_PREVIOUS,
    fork_moves,
    free_piece_moves,
    mate_in_two_moves,
    mating_moves,
    scan_mates,
    scan_motif,
)


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


def scholars_position() -> chess.Board:
    board = chess.Board()
    for san in ("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"):
        board.push_san(san)
    return board


def test_scanning_a_block_reports_only_the_rows_with_a_mate():
    block = pack_board(chess.Board()) + pack_board(scholars_position()) + pack_board(chess.Board())
    assert scan_mates(block) == [(1, [MOVE_TO_INDEX["h5f7"]])]


def test_scanning_nothing_finds_nothing():
    assert scan_mates(b"") == []


# ── Mate in two, free pieces, forks ───────────────────────────────────────

DISCOVERED = "7k/8/5K2/8/8/8/8/Q7 w - - 0 1"  # the king steps aside: check, then Qg7#


def test_a_forced_mate_in_two_by_check_is_found():
    assert mate_in_two_moves(chess.Board(DISCOVERED)) == {"f6f7", "f6g6"}


def test_a_position_with_a_mate_in_one_is_not_a_mate_in_two():
    assert mate_in_two_moves(scholars_position()) == set()


def test_a_quiet_position_has_no_motifs():
    board = chess.Board()
    assert mate_in_two_moves(board) == fork_moves(board) == free_piece_moves(board) == set()


def test_a_knight_fork_of_king_and_queen_is_found():
    assert fork_moves(chess.Board("q3k3/8/8/3N4/8/8/8/4K3 w - - 0 1")) == {"d5c7"}


def test_a_fork_on_a_square_a_bishop_covers_is_not_safe():
    """Nc7+ would fork king and queen, but the bishop on b6 takes the knight."""
    assert fork_moves(chess.Board("q3k3/8/1b6/3N4/8/8/8/4K3 w - - 0 1")) == set()


def test_an_undefended_queen_is_a_free_piece():
    assert free_piece_moves(chess.Board("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1")) == {"d1d5"}


def test_a_queen_swap_is_not_a_free_piece():
    """The black queen is defended by the pawn on e6, so Qxd5 exd5 wins nothing."""
    assert free_piece_moves(chess.Board("4k3/8/4p3/3q4/8/8/8/3QK3 w - - 0 1")) == set()


def test_a_pawn_taking_a_defended_queen_is_a_free_piece():
    assert free_piece_moves(chess.Board("4k3/8/4p3/3q4/2P5/8/8/4K3 w - - 0 1")) == {"c4d5"}


def test_taking_a_pawn_is_not_a_free_piece():
    assert free_piece_moves(chess.Board("4k3/8/8/3p4/8/8/8/3QK3 w - - 0 1")) == set()


def test_the_board_is_left_as_it_was_by_every_motif():
    fens = (DISCOVERED, "q3k3/8/8/3N4/8/8/8/4K3 w - - 0 1", "4k3/8/4p3/3q4/2P5/8/8/4K3 w - - 0 1")
    for fen in fens:
        board = chess.Board(fen)
        for find in (mate_in_two_moves, fork_moves, free_piece_moves):
            find(board)
        assert board.fen() == fen


def test_scanning_for_a_chosen_motif_lists_only_its_rows():
    fork = chess.Board("q3k3/8/8/3N4/8/8/8/4K3 w - - 0 1")
    block = pack_board(chess.Board()) + pack_board(fork)
    assert scan_motif(block, "fork") == [(1, [MOVE_TO_INDEX["d5c7"]])]
    assert scan_motif(block, "mate_in_one") == []


def test_the_default_motif_is_mate_in_one():
    block = pack_board(chess.Board()) + pack_board(scholars_position())
    assert scan_motif(block) == scan_mates(block)


def test_taking_back_where_the_opponent_just_captured_is_not_a_free_piece():
    knight = chess.Board("4k3/8/8/3n4/8/8/8/3QK3 w - - 0 1")
    assert free_piece_moves(knight) == {"d1d5"}  # without the last move, it counts
    assert free_piece_moves(knight, previous_to=chess.D5) == set()  # a recapture
    assert free_piece_moves(knight, previous_to=chess.A1) == {"d1d5"}  # an unrelated move


def test_the_scan_uses_the_previous_move_for_free_pieces_only():
    knight = chess.Board("4k3/8/8/3n4/8/8/8/3QK3 w - - 0 1")
    block = pack_board(knight)
    assert scan_motif(block, "free_piece", bytes([chess.D5])) == []
    assert scan_motif(block, "free_piece", bytes([NO_PREVIOUS])) == [(0, [MOVE_TO_INDEX["d1d5"]])]
    assert scan_motif(block, "free_piece") == [(0, [MOVE_TO_INDEX["d1d5"]])]
