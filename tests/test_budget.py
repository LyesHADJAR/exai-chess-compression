"""The released artifact size is a design constraint, so it is tested."""

from dataclasses import replace

from chessdt.constants import (
    N_MOVES,
    POSITION_TOKENS,
    PRESETS,
    REFERENCE,
    SIZE_CAP_BYTES,
    SPLIT_FRACTIONS,
)


def test_parameter_count_stays_near_target():
    """Capacity is deliberately tight; drifting off target changes the challenge."""
    params = REFERENCE.parameter_count()
    assert 11_000_000 <= params <= 11_600_000, f"{params:,} parameters is off target"


def test_artifact_fits_the_announced_size():
    """Competitors are told the artifact is about 43 MB. Keep it true."""
    mib = REFERENCE.artifact_bytes() / 1024 / 1024
    assert 42.0 <= mib <= 44.0, f"artifact is {mib:.1f} MiB"


def test_stream_length_matches_context():
    assert REFERENCE.stream_tokens == REFERENCE.context_plies * 3


def test_attention_dims_divide_evenly():
    assert REFERENCE.dim % REFERENCE.heads == 0
    assert REFERENCE.enc_dim % REFERENCE.enc_heads == 0


def test_splits_sum_to_one():
    assert abs(sum(SPLIT_FRACTIONS.values()) - 1.0) < 1e-9


def test_encoding_constants_are_the_agreed_values():
    assert POSITION_TOKENS == 77
    assert N_MOVES == 1968


def test_every_preset_fits_the_size_cap_with_the_options_on():
    """The widest option (8 pooled vectors) adds parameters; the cap must still hold."""
    for name, cfg in PRESETS.items():
        loaded = replace(cfg, summary_tokens=8, geometry=True, cond_dropout=0.1)
        assert loaded.artifact_bytes() <= SIZE_CAP_BYTES, f"{name} is {loaded.artifact_bytes()}"


def test_the_ladder_actually_climbs():
    sizes = [PRESETS[n].parameter_count() for n in ("v1", "m18", "m25")]
    assert sizes == sorted(sizes) and sizes[0] < 12e6 < 17e6 < sizes[1] < 19e6 < 24e6 < sizes[2]
