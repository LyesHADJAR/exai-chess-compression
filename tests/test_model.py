"""The reference model, and every v2 option.

The test that matters most here is the readout. The previous iteration of this
project read its prediction from the action token, the model learned to copy
its own input, and nobody noticed until it was trained. These tests fail if
that mistake is reintroduced, one of them checks that they would, and all of
them run against every option, since each option changes what a token can see.
"""

import pytest
import torch

from chessdt.constants import N_MOVES, REFERENCE, ModelConfig
from chessdt.model import ChessDT, load_checkpoint, ply_block_mask, save_checkpoint

TINY = dict(
    enc_dim=32, enc_layers=1, enc_heads=2, enc_ff=64,
    dim=48, layers=2, heads=2, ff=64, context_plies=5, dropout=0.0,
)  # fmt: skip

#: Every option on its own, and all of them together.
CONFIGS = {
    "reference": {},
    "pooled x4": dict(summary_tokens=4),
    "full detail": dict(full_detail=True),
    "geometry": dict(geometry=True),
    "conditioning dropout": dict(cond_dropout=0.5),
    "everything": dict(summary_tokens=3, geometry=True, cond_dropout=0.3),
    "full detail + geometry": dict(full_detail=True, geometry=True),
    "no encoder blocks": dict(full_detail=True, enc_layers=0),
}


def build(name: str) -> ChessDT:
    torch.manual_seed(0)
    return ChessDT(ModelConfig(**{**TINY, **CONFIGS[name]})).eval()


@pytest.fixture(params=CONFIGS)
def model(request):
    return build(request.param)


def inputs(plies: int = 5, batch: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randint(0, 13, (batch, plies, 64), generator=g),
        torch.randint(0, 2, (batch, plies), generator=g),
        torch.randn(batch, plies, 2, generator=g),
        torch.randint(0, N_MOVES, (batch, plies), generator=g),
    ]


# ── Size ──────────────────────────────────────────────────────────────────


def test_the_reference_is_exactly_what_v1_was():
    assert sum(p.numel() for p in ChessDT(REFERENCE).parameters()) == 11_284_656


@pytest.mark.parametrize("name", CONFIGS)
def test_built_model_matches_the_declared_parameter_count(name):
    cfg = ModelConfig(**{**TINY, **CONFIGS[name]})
    assert sum(p.numel() for p in ChessDT(cfg).parameters()) == cfg.parameter_count()


def test_the_reference_size_is_matched_for_the_full_sized_options_too():
    for options in (dict(summary_tokens=8), dict(geometry=True), dict(cond_dropout=0.1)):
        cfg = ModelConfig(**options)
        assert sum(p.numel() for p in ChessDT(cfg).parameters()) == cfg.parameter_count()


# ── Shape ─────────────────────────────────────────────────────────────────


def test_output_is_one_distribution_per_ply(model):
    assert model(*inputs()).shape == (3, 5, N_MOVES)


def test_stream_length_follows_the_options():
    assert ModelConfig(**TINY).stream_tokens == 5 * 3
    assert ModelConfig(**TINY, summary_tokens=4).stream_tokens == 5 * 6
    assert ModelConfig(**TINY, full_detail=True).stream_tokens == 5 * 66


def test_a_shorter_window_is_accepted_and_a_longer_one_refused(model):
    assert model(*inputs(plies=1)).shape == (3, 1, N_MOVES)
    with pytest.raises(ValueError, match="exceeds the context"):
        model(*inputs(plies=6))


# ── The leak ──────────────────────────────────────────────────────────────


def test_the_move_being_predicted_is_not_visible(model):
    """Changing the move played at a ply must not change its own prediction."""
    codes, side, cond, moves = inputs()
    before = model(codes, side, cond, moves)

    for ply in range(moves.shape[1]):
        changed = moves.clone()
        changed[:, ply] = (changed[:, ply] + 1) % N_MOVES
        after = model(codes, side, cond, changed)
        assert torch.equal(before[:, : ply + 1], after[:, : ply + 1]), f"leak at ply {ply}"


def test_earlier_moves_do_inform_later_predictions(model):
    """The guard above must not pass merely because moves are ignored."""
    codes, side, cond, moves = inputs()
    changed = moves.clone()
    changed[:, 0] = (changed[:, 0] + 1) % N_MOVES
    before = model(codes, side, cond, moves)
    after = model(codes, side, cond, changed)
    assert not torch.equal(before[:, 1:], after[:, 1:])


def test_reading_the_move_token_would_be_caught(model, monkeypatch):
    """Reintroduce the old bug and confirm the leak test would have failed."""
    monkeypatch.setattr(model, "readout_offset", model.state_tokens + 1)
    codes, side, cond, moves = inputs()
    changed = moves.clone()
    changed[:, -1] = (changed[:, -1] + 1) % N_MOVES
    before = model(codes, side, cond, moves)
    after = model(codes, side, cond, changed)
    assert not torch.equal(before[:, -1], after[:, -1])


def test_no_ply_sees_the_future(model):
    codes, side, cond, moves = inputs()
    before = model(codes, side, cond, moves)

    later_codes = codes.clone()
    later_codes[:, 3:] = (later_codes[:, 3:] + 1) % 13
    later_cond = cond.clone()
    later_cond[:, 3:] += 1.0
    later_moves = moves.clone()
    later_moves[:, 3:] = (later_moves[:, 3:] + 1) % N_MOVES

    after = model(later_codes, side, later_cond, later_moves)
    assert torch.equal(before[:, :3], after[:, :3])


def test_the_block_mask_hides_the_move_from_every_token_of_its_own_ply():
    plies, tokens = 3, 4
    mask = ply_block_mask(plies, tokens)
    width = tokens + 2
    for ply in range(plies):
        move = ply * width + tokens + 1
        own = range(ply * width, ply * width + tokens + 1)
        assert not mask[list(own), move].any(), "a position token can see its own ply's move"
        assert mask[move, list(own)].all() and mask[move, move]
    index = torch.arange(plies * width)
    later = (index[None, :] // width) > (index[:, None] // width)
    assert not (mask & later).any(), "a token can see a later ply"


def test_positions_see_each_other_both_ways_within_a_ply():
    mask = ply_block_mask(2, 4)
    assert mask[1, 4] and mask[4, 1], "squares of one position must attend in both directions"


# ── Inputs reach the output ───────────────────────────────────────────────


def test_the_board_and_the_condition_both_reach_the_prediction(model):
    codes, side, cond, moves = inputs()
    before = model(codes, side, cond, moves)
    assert not torch.equal(before, model((codes + 1) % 13, side, cond, moves))
    assert not torch.equal(before, model(codes, side, cond + 1.0, moves))
    assert not torch.equal(before, model(codes, 1 - side, cond, moves))


def test_history_free_prediction_matches_the_first_ply_of_a_window(model):
    """A puzzle is scored as a one-ply window; that must equal ply 0 of any window."""
    codes, side, cond, moves = inputs()
    full = model(codes, side, cond, moves)[:, 0]
    alone = model(codes[:, :1], side[:, :1], cond[:, :1], moves[:, :1])[:, 0]
    assert torch.allclose(full, alone, atol=1e-5)


# ── Geometry ──────────────────────────────────────────────────────────────


def test_a_square_is_the_sum_of_its_file_rank_and_diagonals():
    model = build("geometry")
    squares = model.encoder.square_position()
    enc = model.encoder
    e4 = 3 * 8 + 4  # file e (index 4), rank 4 (index 3)
    expected = (
        enc.file_embed.weight[4]
        + enc.rank_embed.weight[3]
        + enc.diag_embed.weight[3 - 4 + 7]
        + enc.anti_embed.weight[3 + 4]
    )
    assert torch.allclose(squares[e4], expected)


def test_squares_on_one_diagonal_share_their_diagonal_embedding():
    enc = build("geometry").encoder
    assert enc.diags[0] == enc.diags[9] == enc.diags[63]  # a1, b2, h8
    assert enc.antis[7] == enc.antis[14] == enc.antis[56]  # h1, g2, a8


def test_relative_bias_depends_only_on_the_offset_between_squares():
    from chessdt.model import relative_square_index

    index = relative_square_index()
    assert index[0, 9] == index[10, 19] == index[27, 36]  # all one step up and right
    assert index[0, 9] != index[0, 10]
    assert index.max() == 224 and index.min() == 0


# ── Conditioning dropout ──────────────────────────────────────────────────


def test_a_model_with_conditioning_dropout_can_be_asked_without_conditioning():
    model = build("conditioning dropout")
    codes, side, _, moves = inputs()
    unknown = model(codes, side, None, moves)
    assert unknown.shape == (3, 5, N_MOVES)
    assert not torch.equal(unknown, model(codes, side, torch.zeros(3, 5, 2), moves))


def test_a_model_without_it_refuses():
    codes, side, _, moves = inputs()
    with pytest.raises(ValueError, match="without conditioning dropout"):
        build("reference")(codes, side, None, moves)


def test_dropout_only_acts_in_training():
    model = build("conditioning dropout")
    codes, side, cond, moves = inputs()
    assert torch.equal(model(codes, side, cond, moves), model(codes, side, cond, moves))
    model.train()
    seen = {tuple(model(codes, side, cond, moves).flatten()[:4].tolist()) for _ in range(6)}
    assert len(seen) > 1, "training-mode dropout should vary the output"


# ── Checkpoints ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["reference", "everything"])
def test_checkpoint_round_trips_without_pickled_classes(name, tmp_path):
    model = build(name)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, step=7)

    restored, payload = load_checkpoint(path)
    assert payload["step"] == 7
    assert isinstance(payload["config"], dict)
    assert torch.equal(model(*inputs()), restored(*inputs()))


def test_a_checkpoint_saved_before_the_options_existed_still_loads(tmp_path):
    """v1 stored a config with none of the new fields; they must default to v1."""
    model = build("reference")
    config = {k: v for k, v in vars(model.cfg).items()}
    for option in ("summary_tokens", "full_detail", "geometry", "cond_dropout"):
        config.pop(option)
    torch.save({"model": model.state_dict(), "config": config, "step": 1}, tmp_path / "old.pt")

    restored, _ = load_checkpoint(tmp_path / "old.pt")
    assert torch.equal(model(*inputs()), restored(*inputs()))
