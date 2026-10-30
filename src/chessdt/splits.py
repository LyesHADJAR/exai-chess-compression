"""
Assigning positions to train, validation and test.

Two guards, and both are needed. Splitting by game stops consecutive positions
from the same game landing on both sides of the boundary. Dropping held-out
positions that also occur in training stops the evaluation measuring memorised
opening theory — the first ten to fifteen plies of almost any game appear
verbatim in thousands of others.

Neither guard is worth anything unsupervised, so `verify` re-checks the result
from scratch and raises rather than warns.
"""

import numpy as np

from .constants import SPLIT_FRACTIONS
from .encoding import POSITION_IDENTITY_BYTES

TRAIN = np.uint8(0)
VAL = np.uint8(1)
TEST = np.uint8(2)
DROPPED = np.uint8(3)

LABEL_NAMES = {TRAIN: "train", VAL: "val", TEST: "test", DROPPED: "dropped"}

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = np.uint64(0x100000001B3)


def fnv1a_64(rows: np.ndarray) -> np.ndarray:
    """Hash each row of a 2-D uint8 array to a uint64.

    FNV-1a, vectorised across rows so 20M+ of them cost seconds rather than
    minutes. At 64 bits over tens of millions of distinct values the chance of
    any collision is on the order of 1e-5, which is well below the rate at
    which we would notice one.
    """
    if rows.ndim != 2:
        raise ValueError(f"expected a 2-D array of bytes, got shape {rows.shape}")

    digest = np.full(rows.shape[0], _FNV_OFFSET, dtype=np.uint64)
    for column in range(rows.shape[1]):
        digest ^= rows[:, column].astype(np.uint64)
        digest *= _FNV_PRIME
    return digest


def position_keys(positions: np.ndarray) -> np.ndarray:
    """Identity key per position, ignoring the move counters.

    `positions` is the (N, BOARD_RECORD_BYTES) corpus.
    """
    return fnv1a_64(positions[:, :POSITION_IDENTITY_BYTES])


def assign_games(game_ids: np.ndarray, fractions: dict[str, float] | None = None) -> np.ndarray:
    """Place each game in a split by hashing its ID.

    Hashing rather than storing the assignment means the split is reproducible
    from the dump alone: no split file to pass around, lose, or disagree about.
    """
    fractions = fractions or SPLIT_FRACTIONS
    # ascontiguousarray: a sliced or reversed view cannot be reinterpreted as
    # bytes, and callers pass whatever slice of the index they happen to hold.
    rows = np.ascontiguousarray(game_ids).view(np.uint8).reshape(len(game_ids), -1)
    position = (fnv1a_64(rows) >> np.uint64(11)) / float(1 << 53)

    train_end = fractions["train"]
    val_end = train_end + fractions["val"]

    labels = np.full(len(game_ids), TEST, dtype=np.uint8)
    labels[position < val_end] = VAL
    labels[position < train_end] = TRAIN
    return labels


def expand_to_positions(
    game_labels: np.ndarray, offsets: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    """Spread per-game labels across the positions of each game.

    Relies on games being written contiguously and in order, which the
    extractor guarantees; the assertion is here because silently violating it
    would misalign every label.
    """
    expected = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    if not np.array_equal(offsets, expected):
        raise ValueError("game offsets are not contiguous; the corpus is misaligned")
    return np.repeat(game_labels, lengths)


def drop_leaked(labels: np.ndarray, keys: np.ndarray) -> int:
    """Mark held-out positions that also occur in training as DROPPED.

    Modifies `labels` in place and returns how many were dropped.
    """
    train_keys = keys[labels == TRAIN]
    held = np.flatnonzero((labels == VAL) | (labels == TEST))
    if not len(held) or not len(train_keys):
        return 0

    leaked = held[np.isin(keys[held], train_keys)]
    labels[leaked] = DROPPED
    return len(leaked)


def verify(labels: np.ndarray, keys: np.ndarray) -> None:
    """Re-check the split from scratch. Raises on any surviving leak.

    Deliberately independent of `drop_leaked`: a dedupe that quietly did
    nothing and a dedupe that worked look identical without this.
    """
    train_keys = keys[labels == TRAIN]
    for label in (VAL, TEST):
        overlap = int(np.isin(keys[labels == label], train_keys).sum())
        if overlap:
            raise AssertionError(
                f"{overlap:,} {LABEL_NAMES[label]} positions still occur in train; "
                "the deduplication did not take effect"
            )

    if not len(train_keys):
        raise AssertionError("the training split is empty")


def summarise(labels: np.ndarray, game_labels: np.ndarray) -> dict[str, dict[str, int]]:
    """Counts per split, for the report."""
    return {
        LABEL_NAMES[label]: {
            "games": int((game_labels == label).sum()),
            "positions": int((labels == label).sum()),
        }
        for label in (TRAIN, VAL, TEST, DROPPED)
    }
