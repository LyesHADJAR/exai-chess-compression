"""Split assignment and leak removal.

The deduplication is the step whose failure is invisible: a corpus that was
never deduped and one that was look identical from the outside. So these tests
check that the guard actually bites, not merely that it runs.
"""

import chess
import numpy as np
import pytest

from chessdt.encoding import BOARD_RECORD_BYTES, pack_board
from chessdt.splits import (
    DROPPED,
    TEST,
    TRAIN,
    VAL,
    assign_games,
    drop_leaked,
    expand_to_positions,
    fnv1a_64,
    position_keys,
    verify,
)


def records(*boards: chess.Board) -> np.ndarray:
    """Pack boards into a corpus-shaped array."""
    return np.frombuffer(b"".join(pack_board(b) for b in boards), dtype=np.uint8).reshape(
        -1, BOARD_RECORD_BYTES
    )


def ids(count: int) -> np.ndarray:
    return np.asarray([f"game{i:07d}" for i in range(count)], dtype="S12")


# ── Hashing ───────────────────────────────────────────────────────────────


def test_hash_is_stable_across_calls():
    rows = np.arange(40, dtype=np.uint8).reshape(4, 10)
    assert np.array_equal(fnv1a_64(rows), fnv1a_64(rows))


def test_hash_separates_different_rows():
    rows = np.array([[1, 2, 3], [1, 2, 4], [9, 9, 9]], dtype=np.uint8)
    assert len(set(fnv1a_64(rows).tolist())) == 3


def test_hash_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        fnv1a_64(np.zeros(10, dtype=np.uint8))


def test_position_identity_ignores_the_halfmove_clock():
    """The same board at a different move number is the same position."""
    early, late = chess.Board(), chess.Board()
    late.halfmove_clock = 42
    assert position_keys(records(early))[0] == position_keys(records(late))[0]


def test_position_identity_still_sees_the_board():
    moved = chess.Board()
    moved.push_san("e4")
    assert position_keys(records(chess.Board()))[0] != position_keys(records(moved))[0]


def test_position_identity_sees_castling_rights():
    with_rights = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    without = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    assert position_keys(records(with_rights))[0] != position_keys(records(without))[0]


# ── Game assignment ───────────────────────────────────────────────────────


def test_assignment_is_deterministic():
    game_ids = ids(500)
    assert np.array_equal(assign_games(game_ids), assign_games(game_ids))


def test_assignment_lands_near_the_requested_fractions():
    labels = assign_games(ids(20_000))
    train = (labels == TRAIN).mean()
    assert 0.88 <= train <= 0.92, train
    assert 0.03 <= (labels == VAL).mean() <= 0.07
    assert 0.03 <= (labels == TEST).mean() <= 0.07


def test_assignment_depends_only_on_the_id():
    """Reordering the games must not move any of them between splits."""
    game_ids = ids(1_000)
    first = dict(zip(game_ids.tolist(), assign_games(game_ids).tolist(), strict=True))

    shuffled = game_ids[::-1]
    second = dict(zip(shuffled.tolist(), assign_games(shuffled).tolist(), strict=True))

    assert first == second


# ── Expansion ─────────────────────────────────────────────────────────────


def test_expansion_spreads_labels_across_each_game():
    labels = expand_to_positions(
        np.array([TRAIN, TEST], dtype=np.uint8),
        offsets=np.array([0, 3]),
        lengths=np.array([3, 2]),
    )
    assert labels.tolist() == [TRAIN, TRAIN, TRAIN, TEST, TEST]


def test_expansion_rejects_a_misaligned_corpus():
    with pytest.raises(ValueError, match="contiguous"):
        expand_to_positions(
            np.array([TRAIN, TEST], dtype=np.uint8),
            offsets=np.array([0, 99]),
            lengths=np.array([3, 2]),
        )


# ── Leak removal ──────────────────────────────────────────────────────────


def shared_corpus():
    """Two positions, the second shared between train and test."""
    start = chess.Board()
    after = chess.Board()
    after.push_san("e4")
    # train: start, after.   test: after (a leak), start-with-a-different-move.
    other = chess.Board()
    other.push_san("d4")
    corpus = records(start, after, after, other)
    labels = np.array([TRAIN, TRAIN, TEST, TEST], dtype=np.uint8)
    return corpus, labels


def test_a_shared_position_is_dropped_from_the_held_out_split():
    corpus, labels = shared_corpus()
    keys = position_keys(corpus)

    assert drop_leaked(labels, keys) == 1
    assert labels.tolist() == [TRAIN, TRAIN, DROPPED, TEST]


def test_training_positions_are_never_dropped():
    corpus, labels = shared_corpus()
    drop_leaked(labels, position_keys(corpus))
    assert (labels[:2] == TRAIN).all()


def test_verify_passes_once_the_leak_is_gone():
    corpus, labels = shared_corpus()
    keys = position_keys(corpus)
    drop_leaked(labels, keys)
    verify(labels, keys)  # must not raise


def test_verify_catches_a_dedupe_that_did_nothing():
    """The failure this whole module exists to prevent."""
    corpus, labels = shared_corpus()
    with pytest.raises(AssertionError, match="did not take effect"):
        verify(labels, position_keys(corpus))


def test_verify_rejects_an_empty_training_split():
    corpus, _ = shared_corpus()
    labels = np.full(len(corpus), TEST, dtype=np.uint8)
    with pytest.raises(AssertionError, match="empty"):
        verify(labels, position_keys(corpus))
