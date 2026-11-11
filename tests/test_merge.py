"""Merging corpora, and the rating floor that makes a second corpus worth having."""

import sys
from pathlib import Path

import numpy as np
import pytest

from chessdt.lichess import Tally, classify
from tests.test_data import write_corpus

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from merge_corpora import merge  # noqa: E402


def part(tmp_path, name, games):
    directory = tmp_path / name
    directory.mkdir()
    write_corpus(directory, games)
    ids = np.asarray([f"{name}{i}" for i in range(len(games))], dtype="S12")
    meta = dict(np.load(directory / "games.npz"))
    meta["game_id"] = ids
    np.savez(directory / "games.npz", **meta)
    return directory


GAME_A = (["e4", "e5", "Nf3"], 1, 2000, 1900, 0)
GAME_B = (["d4", "d5", "c4", "e6"], -1, 1500, 1600, 0)


def test_games_keep_their_plies_after_merging(tmp_path):
    a = part(tmp_path, "a", [GAME_A, GAME_B])
    b = part(tmp_path, "b", [GAME_B])
    totals = merge([a, b], tmp_path / "out")
    assert totals == {"games": 3, "positions": 3 + 4 + 4}

    meta = np.load(tmp_path / "out" / "games.npz")
    assert meta["offset"].tolist() == [0, 3, 7]
    assert meta["length"].tolist() == [3, 4, 4]
    assert meta["white_elo"].tolist() == [2000, 1500, 1500]


def test_merged_positions_are_the_inputs_end_to_end(tmp_path):
    a = part(tmp_path, "a", [GAME_A])
    b = part(tmp_path, "b", [GAME_B])
    merge([a, b], tmp_path / "out")
    joined = (tmp_path / "out" / "positions.bin").read_bytes()
    assert joined == (a / "positions.bin").read_bytes() + (b / "positions.bin").read_bytes()
    assert (tmp_path / "out" / "moves.bin").stat().st_size == 2 * (3 + 4)


def test_a_game_in_two_corpora_is_refused(tmp_path):
    a = part(tmp_path, "a", [GAME_A])
    with pytest.raises(ValueError, match="more than one corpus"):
        merge([a, a], tmp_path / "out")


def test_a_corrupt_corpus_is_refused(tmp_path):
    a = part(tmp_path, "a", [GAME_A])
    (a / "positions.bin").write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError, match="inconsistent"):
        merge([a], tmp_path / "out")


# ── the rating floor ──────────────────────────────────────────────────────

WEAK = {"WhiteElo": "1200", "BlackElo": "1100", "TimeControl": "300+0", "Termination": "Normal"}


def test_a_lower_floor_lets_weaker_games_through():
    assert classify(WEAK, Tally()) is False  # the 1800 default
    assert classify(WEAK, Tally(), min_elo=1000) is True


def test_the_floor_applies_to_both_players():
    assert classify({**WEAK, "BlackElo": "900"}, Tally(), min_elo=1000) is False
