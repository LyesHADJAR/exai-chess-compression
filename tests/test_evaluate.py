"""Scoring helpers."""

import torch

from chessdt.evaluate import topk_hits


def logits_ranking(*rows):
    """Logits whose ranking is the given order of move indices, best first."""
    out = torch.full((len(rows), 6), -10.0)
    for r, order in enumerate(rows):
        for rank, move in enumerate(order):
            out[r, move] = float(len(order) - rank)
    return out


def test_top1_is_the_first_choice_only():
    logits = logits_ranking([4, 2, 0], [1, 3, 5])
    assert topk_hits(logits, torch.tensor([4, 3]), 1).tolist() == [True, False]


def test_top3_accepts_any_of_the_three_favourites():
    logits = logits_ranking([4, 2, 0], [1, 3, 5])
    assert topk_hits(logits, torch.tensor([0, 3]), 3).tolist() == [True, True]


def test_top3_rejects_the_fourth_choice():
    logits = logits_ranking([4, 2, 0, 5])
    assert topk_hits(logits, torch.tensor([5]), 3).tolist() == [False]


def test_top3_can_never_be_below_top1():
    torch.manual_seed(0)
    logits, target = torch.randn(500, 40), torch.randint(0, 40, (500,))
    assert (topk_hits(logits, target, 3) >= topk_hits(logits, target, 1)).all()
