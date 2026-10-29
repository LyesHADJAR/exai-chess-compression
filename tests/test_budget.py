"""The released artifact size is a design constraint, so it is tested."""

from chessdt.constants import N_MOVES, POSITION_TOKENS, REFERENCE, SPLIT_FRACTIONS


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
