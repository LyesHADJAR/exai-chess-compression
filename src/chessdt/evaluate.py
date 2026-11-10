"""
Scoring a model on held-out positions.

One protocol, used by training and by the evaluation script alike: each
held-out game is tiled with consecutive windows from its first ply, the model
predicts every ply, and only positions carrying the split's own label count.
Positions dropped as leaked still serve as history but are never scored.
"""

import torch
import torch.nn.functional as F

from .data import Batch, Corpus


def autocast(device: str):
    """Mixed precision where it helps, a no-op elsewhere."""
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return torch.autocast("cpu", enabled=False)


def topk_hits(logits: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    """Per row, whether the target is among the k highest-scoring moves.

    Top-1 asks whether the model's first choice is the move that was played.
    Top-3 asks whether the played move is one of its three favourites, which
    is the fairer measure of behaviour cloning: in most positions a human
    chooses between a few reasonable moves, and a model that ranks the played
    one second has still understood the position.
    """
    return (logits.topk(k, dim=-1).indices == target.unsqueeze(-1)).any(dim=-1)


@torch.no_grad()
def predict(model, batch: Batch, device: str) -> torch.Tensor:
    """Logits for every ply of a batch, as float32."""
    with autocast(device):
        return model(*batch.inputs()).float()


@torch.no_grad()
def score(
    model,
    corpus: Corpus,
    starts: torch.Tensor,
    plies: int,
    split: int,
    batch_size: int = 512,
    mutate=None,
) -> dict:
    """Loss, top-1 and top-3 accuracy over the positions of `split` in the windows.

    `mutate` optionally rewrites each batch before the forward pass; the
    ablations are built on it.
    """
    model.eval()
    loss_sum, hits, hits3, count = 0.0, 0, 0, 0

    for i in range(0, len(starts), batch_size):
        batch = corpus.window(starts[i : i + batch_size], plies)
        if mutate is not None:
            batch = mutate(batch)
        keep = batch.valid & (batch.labels == split)

        logits = predict(model, batch, corpus.device)[keep]
        target = batch.moves[keep]
        loss_sum += F.cross_entropy(logits, target, reduction="sum").item()
        hits += topk_hits(logits, target, 1).sum().item()
        hits3 += topk_hits(logits, target, 3).sum().item()
        count += int(keep.sum())

    return {
        "loss": loss_sum / max(count, 1),
        "top1": hits / max(count, 1),
        "top3": hits3 / max(count, 1),
        "positions": count,
    }


@torch.no_grad()
def compare(
    model,
    corpus: Corpus,
    starts: torch.Tensor,
    plies: int,
    split: int,
    mutate,
    batch_size: int = 512,
) -> dict:
    """How much an intervention on the inputs moves the predictions.

    Runs every window twice, as it is and after `mutate`, and reports the
    share of positions whose chosen move changes. A channel the model ignores
    scores zero here however the input is mangled, which is the whole point:
    this is the measurement that caught the previous model copying its input.
    """
    model.eval()
    flips, hits, count = 0, 0, 0

    for i in range(0, len(starts), batch_size):
        batch = corpus.window(starts[i : i + batch_size], plies)
        keep = batch.valid & (batch.labels == split)

        before = predict(model, batch, corpus.device).argmax(-1)
        after = predict(model, mutate(batch), corpus.device).argmax(-1).reshape(before.shape)

        flips += (before[keep] != after[keep]).sum().item()
        hits += (after[keep] == batch.moves[keep]).sum().item()
        count += int(keep.sum())

    return {"changed": flips / max(count, 1), "top1": hits / max(count, 1), "positions": count}


def alone(batch: Batch) -> Batch:
    """Every ply as its own one-ply window: the same positions, no history."""
    return Batch(
        codes=batch.codes.flatten(0, 1).unsqueeze(1),
        side=batch.side.flatten().unsqueeze(1),
        cond=batch.cond.flatten(0, 1).unsqueeze(1),
        moves=batch.moves.flatten().unsqueeze(1),
        valid=batch.valid.flatten().unsqueeze(1),
        labels=batch.labels.flatten().unsqueeze(1),
    )
