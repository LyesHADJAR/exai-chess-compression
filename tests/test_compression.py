"""The reference compression methods."""

import pytest
import torch

from chessdt.compression import (
    GROUPS,
    group_of,
    low_rank_,
    low_rank_params,
    parameter_groups,
    prune_,
    quantize_,
    rank_for,
    weight_matrices,
)
from chessdt.constants import REFERENCE, ModelConfig
from chessdt.model import ChessDT


def matrix(rows=16, cols=64, seed=0):
    return torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed))


# ── Quantisation ──────────────────────────────────────────────────────────


def test_int8_is_close_and_int4_is_coarser():
    original = matrix()
    errors = {}
    for bits in (8, 4, 2):
        w = original.clone()
        quantize_(w, bits)
        errors[bits] = (w - original).abs().mean().item()
    assert errors[8] < errors[4] < errors[2]
    assert errors[8] < 0.01


def test_quantising_leaves_at_most_2_to_the_bits_levels_per_row():
    w = matrix()
    quantize_(w, 4)
    assert all(len(row.unique()) <= 16 for row in w)


def test_quantisation_is_idempotent():
    w = matrix()
    quantize_(w, 4)
    again = w.clone()
    quantize_(again, 4)
    assert torch.allclose(w, again, atol=1e-6)


def test_sixteen_bits_is_left_alone():
    w = matrix()
    before = w.clone()
    quantize_(w, 16)
    assert torch.equal(w, before)


def test_each_row_has_its_own_scale():
    """A row with tiny values must not be flattened by a row with huge ones."""
    w = torch.stack([torch.linspace(-1e-3, 1e-3, 64), torch.linspace(-100, 100, 64)])
    quantize_(w, 8)
    assert w[0].abs().max() > 5e-4


# ── Pruning ───────────────────────────────────────────────────────────────


def test_pruning_zeroes_exactly_the_requested_fraction():
    ws = [matrix(seed=1), matrix(seed=2)]
    total = sum(w.numel() for w in ws)
    assert prune_(ws, 0.25) == int(0.25 * total)
    assert sum((w == 0).sum().item() for w in ws) == int(0.25 * total)


def test_magnitude_pruning_removes_the_smallest_weights():
    w = torch.tensor([[5.0, -0.1, 3.0, 0.2, -4.0, 0.05, 2.0, 1.0]])
    prune_([w], 0.375)
    assert (w[0] == 0).nonzero().flatten().tolist() == [1, 3, 5]


def test_the_threshold_is_global_across_matrices():
    small, large = torch.full((1, 8), 0.1), torch.full((1, 8), 10.0)
    prune_([small, large], 0.5)
    assert (small == 0).all() and (large != 0).all()


def test_random_pruning_hits_the_same_count_but_not_the_same_weights():
    a, b = matrix(seed=3), matrix(seed=3)
    assert prune_([a], 0.3) == prune_([b], 0.3, random=True)
    assert not torch.equal(a == 0, b == 0)


def test_pruning_nothing_changes_nothing():
    w = matrix()
    before = w.clone()
    assert prune_([w], 0.0) == 0
    assert torch.equal(w, before)


# ── Low rank ──────────────────────────────────────────────────────────────


def test_low_rank_reduces_the_rank_and_keeps_the_shape():
    w = matrix(32, 128)
    assert low_rank_(w, 0.25) == 8
    assert w.shape == (32, 128) and torch.linalg.matrix_rank(w).item() == 8


def test_a_factorisation_that_would_not_save_bytes_is_not_applied():
    """Two rank-288 factors of a 384x384 matrix are bigger than the matrix itself."""
    w = matrix(48, 48)
    before = w.clone()
    assert rank_for((48, 48), 0.75) is None
    assert low_rank_(w, 0.75) is None
    assert torch.equal(w, before)


def test_the_rank_threshold_is_where_the_two_factors_stop_being_smaller():
    m, n = 384, 1024
    breakeven = m * n // (m + n)
    assert rank_for((m, n), (breakeven - 2) / min(m, n)) is not None
    assert rank_for((m, n), (breakeven + 2) / min(m, n)) is None


def test_a_rank_r_factorisation_stores_r_times_the_two_sides():
    assert low_rank_params((384, 1024), 96) == 96 * (384 + 1024)


# ── Groups ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "group"),
    [
        ("encoder.blocks.2.attn.qkv.weight", "board encoder: attention"),
        ("encoder.blocks.0.ffn.w1.weight", "board encoder: feed-forward"),
        ("encoder.code_embed.weight", "board encoder: embeddings"),
        ("encoder.square_embed.weight", "board encoder: embeddings"),
        ("encoder.proj.weight", "projections"),
        ("cond_embed.weight", "projections"),
        ("blocks.3.attn.proj.weight", "backbone: attention"),
        ("blocks.1.ffn.w2.weight", "backbone: feed-forward"),
        ("move_embed.weight", "move embedding"),
        ("ply_embed.weight", "ply embedding"),
        ("head.weight", "policy head"),
    ],
)
def test_weights_are_grouped_by_where_they_live(name, group):
    assert group_of(name) == group


def test_every_weight_in_the_reference_model_lands_in_a_known_group():
    model = ChessDT(REFERENCE)
    sizes = parameter_groups(model)
    assert set(sizes) <= set(GROUPS)
    assert "other" not in sizes
    # Matrices only: norms and biases are not compressed.
    total = sum(p.numel() for _, p in weight_matrices(model))
    assert sum(sizes.values()) == total


def test_a_single_group_can_be_selected():
    small = ModelConfig(
        enc_dim=32, enc_layers=1, enc_heads=2, enc_ff=64,
        dim=48, layers=1, heads=2, ff=64, context_plies=4,
    )  # fmt: skip
    model = ChessDT(small)
    names = [n for n, _ in weight_matrices(model, only="policy head")]
    assert names == ["head.weight"]
