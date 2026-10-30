"""Move vocabulary and board packing.

The packed record format is frozen once a corpus is encoded and a model is
trained on it, so these tests guard the format rather than merely exercising it.
"""

import chess
import pytest

from chessdt.constants import N_MOVES
from chessdt.encoding import (
    BOARD_RECORD_BYTES,
    MOVE_TO_INDEX,
    MOVE_VOCABULARY,
    pack_board,
    unpack_board,
)


def play(*sans: str) -> chess.Board:
    """A board after the given moves in SAN."""
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


# ── Move vocabulary ───────────────────────────────────────────────────────


def test_vocabulary_is_the_expected_size():
    assert len(MOVE_VOCABULARY) == N_MOVES


def test_vocabulary_is_sorted_and_unique():
    assert list(MOVE_VOCABULARY) == sorted(MOVE_VOCABULARY)
    assert len(set(MOVE_VOCABULARY)) == len(MOVE_VOCABULARY)


def test_index_map_is_consistent_with_the_vocabulary():
    for index, uci in enumerate(MOVE_VOCABULARY):
        assert MOVE_TO_INDEX[uci] == index


def test_vocabulary_covers_ordinary_moves_and_promotions():
    for uci in ("e2e4", "g1f3", "e1g1", "a7a8q", "a7b8n", "h2h1r", "b7a8b"):
        assert uci in MOVE_TO_INDEX, uci


def test_vocabulary_excludes_impossible_geometry():
    # No piece moves two files and three ranks, and a move cannot stay put.
    assert "a1c4" not in MOVE_TO_INDEX
    assert "e4e4" not in MOVE_TO_INDEX


def test_every_legal_move_of_a_real_game_is_in_the_vocabulary():
    board = chess.Board()
    for san in ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"):
        for move in board.legal_moves:
            assert move.uci() in MOVE_TO_INDEX, move.uci()
        board.push_san(san)


# ── Packed records ────────────────────────────────────────────────────────


def test_record_is_fixed_width():
    assert len(pack_board(chess.Board())) == BOARD_RECORD_BYTES


@pytest.mark.parametrize(
    ("name", "board"),
    [
        ("start", chess.Board()),
        ("after e4", play("e4")),
        ("sicilian", play("e4", "c5", "Nf3", "d6", "d4", "cxd4")),
        ("castled", play("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O")),
        ("empty", chess.Board.empty()),
        ("promotion ready", chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")),
        ("en passant", chess.Board("8/8/8/3pP3/8/8/8/K6k w - d6 0 1")),
        ("no castling", chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")),
        ("all castling", chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")),
        ("black to move", chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 7 1")),
    ],
)
def test_round_trip_preserves_the_position(name, board):
    restored = unpack_board(pack_board(board))

    assert restored.piece_map() == board.piece_map(), name
    assert restored.turn == board.turn, name
    assert restored.castling_rights == board.castling_rights, name
    assert restored.ep_square == board.ep_square, name
    assert restored.halfmove_clock == board.halfmove_clock, name


def test_round_trip_survives_a_whole_game():
    board = chess.Board()
    for san in ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "e3", "O-O", "Bd3", "d5"):
        board.push_san(san)
        restored = unpack_board(pack_board(board))
        assert restored.piece_map() == board.piece_map()
        assert restored.turn == board.turn


def test_halfmove_clock_saturates_rather_than_overflowing():
    board = chess.Board()
    board.halfmove_clock = 400
    assert unpack_board(pack_board(board)).halfmove_clock == 255


def test_unpack_rejects_a_wrong_width_record():
    with pytest.raises(ValueError, match="expected"):
        unpack_board(b"\x00" * (BOARD_RECORD_BYTES - 1))
