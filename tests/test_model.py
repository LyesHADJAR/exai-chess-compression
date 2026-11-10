"""The reference model.

The test that matters most here is the readout. The previous iteration of this
project read its prediction from the action token, the model learned to copy
its own input, and nobody noticed until it was trained. These tests fail if
that mistake is reintroduced, and one of them checks that they would.
"""

import pytest
import torch

from chessdt.constants import N_MOVES, REFERENCE, ModelConfig
from chessdt.model import ChessDT, load_checkpoint, save_checkpoint

SMALL = ModelConfig(
    enc_dim=32, enc_layers=1, enc_heads=2, enc_ff=64,
    dim=48, layers=2, heads=2, ff=64, context_plies=6, dropout=0.0,
)  # fmt: skip


def inputs(batch: int = 3, plies: int = 6, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randint(0, 13, (batch, plies, 64), generator=g),
        torch.randint(0, 2, (batch, plies), generator=g),
        torch.randn(batch, plies, 2, generator=g),
        torch.randint(0, N_MOVES, (batch, plies), generator=g),
    ]


@pytest.fixture
def model():
    torch.manual_seed(0)
    return ChessDT(SMALL).eval()


def test_built_model_matches_the_declared_parameter_count():
    built = sum(p.numel() for p in ChessDT(REFERENCE).parameters())
    assert built == REFERENCE.parameter_count()


def test_output_is_one_distribution_per_ply(model):
    assert model(*inputs()).shape == (3, 6, N_MOVES)


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
    monkeypatch.setattr(ChessDT, "READOUT_OFFSET", 2)
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

    after = model(later_codes, side, later_cond, moves)
    assert torch.equal(before[:, :3], after[:, :3])


def test_the_board_and_the_condition_both_reach_the_prediction(model):
    codes, side, cond, moves = inputs()
    before = model(codes, side, cond, moves)
    assert not torch.equal(before, model((codes + 1) % 13, side, cond, moves))
    assert not torch.equal(before, model(codes, side, cond + 1.0, moves))
    assert not torch.equal(before, model(codes, 1 - side, cond, moves))


def test_a_shorter_window_is_accepted_and_a_longer_one_refused(model):
    assert model(*inputs(plies=1)).shape == (3, 1, N_MOVES)
    with pytest.raises(ValueError, match="exceeds the context"):
        model(*inputs(plies=7))


def test_history_free_prediction_matches_the_first_ply_of_a_window(model):
    """A puzzle is scored as a one-ply window; that must equal ply 0 of any window."""
    codes, side, cond, moves = inputs()
    full = model(codes, side, cond, moves)[:, 0]
    alone = model(codes[:, :1], side[:, :1], cond[:, :1], moves[:, :1])[:, 0]
    assert torch.allclose(full, alone, atol=1e-5)


def test_checkpoint_round_trips_without_pickled_classes(model, tmp_path):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, step=7)

    restored, payload = load_checkpoint(path)
    assert payload["step"] == 7
    assert isinstance(payload["config"], dict)
    assert torch.equal(model(*inputs()), restored(*inputs()))
