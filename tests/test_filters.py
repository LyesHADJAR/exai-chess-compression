"""Collection filters, tested without touching the network."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from measure_pass_rate import Tally, classify, initial_clock, parse_elo  # noqa: E402


def headers(**overrides) -> dict:
    """A game that passes every filter, unless an override breaks it."""
    base = {
        "WhiteElo": "2000",
        "BlackElo": "1900",
        "TimeControl": "300+0",
        "Termination": "Normal",
    }
    base.update(overrides)
    return base


def test_initial_clock_reads_the_base_seconds():
    assert initial_clock("600+0") == 600
    assert initial_clock("180+2") == 180


def test_initial_clock_rejects_correspondence_and_junk():
    assert initial_clock("-") is None
    assert initial_clock("") is None
    assert initial_clock(None) is None


def test_parse_elo_rejects_unrated():
    assert parse_elo("1850") == 1850
    assert parse_elo("?") is None
    assert parse_elo(None) is None


def test_a_good_game_passes():
    assert classify(headers(), Tally()) is True


def test_rating_floor_applies_to_both_players():
    assert classify(headers(WhiteElo="1799"), Tally()) is False
    assert classify(headers(BlackElo="1799"), Tally()) is False
    assert classify(headers(WhiteElo="1800", BlackElo="1800"), Tally()) is True


def test_bullet_is_excluded_but_three_minute_blitz_is_not():
    assert classify(headers(TimeControl="60+0"), Tally()) is False
    assert classify(headers(TimeControl="179+0"), Tally()) is False
    # 3+0 sits exactly on the boundary and is one of the most played controls.
    assert classify(headers(TimeControl="180+0"), Tally()) is True


def test_abandoned_games_are_dropped_but_time_forfeits_are_kept():
    assert classify(headers(Termination="Abandoned"), Tally()) is False
    assert classify(headers(Termination="Time forfeit"), Tally()) is True


def test_tally_records_where_games_were_lost():
    tally = Tally()
    classify(headers(), tally)
    classify(headers(WhiteElo="1200"), tally)
    classify(headers(TimeControl="60+0"), tally)

    assert tally.scanned == 3
    assert tally.rated == 3
    assert tally.rating_ok == 2  # the 1200 was dropped here
    assert tally.clock_ok == 1  # the bullet game was dropped here
    assert tally.termination_ok == 1
    assert tally.pass_rate == 1 / 3
