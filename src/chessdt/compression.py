"""
Reference implementations of the standard data-free compression methods.

These are what a competitor would reach for first, applied to the released
weights with no data and no training. They exist for two reasons: gate 3 uses
them to check that the scoring axis separates methods, and they set the
baseline that a clever submission has to beat.

Everything here simulates compression by rewriting weights in place ("fake
quantisation"): the result is a model of the same shape whose weights have
been degraded exactly as the real method would degrade them. Whether the
result then stores in fewer bytes is a separate question, answered by
`stored_bits`.
"""

import torch
import torch.nn as nn

#: Weight matrices are what gets compressed. Norms and biases are a rounding
#: error in the byte count and far more sensitive per byte, so they stay fp32.
COMPRESSIBLE = (nn.Linear, nn.Embedding)

#: The groups sensitivity is reported by, in the order they are listed.
GROUPS = (
    "board encoder: attention",
    "board encoder: feed-forward",
    "board encoder: embeddings",
    "backbone: attention",
    "backbone: feed-forward",
    "projections",
    "move embedding",
    "ply embedding",
    "policy head",
)


def group_of(name: str) -> str:
    """Which sensitivity group a weight belongs to, from its parameter name."""
    if name.startswith("encoder."):
        if ".attn." in name or name.startswith("encoder.pool."):
            return "board encoder: attention"
        if ".ffn." in name:
            return "board encoder: feed-forward"
        if name.startswith("encoder.proj"):
            return "projections"
        return "board encoder: embeddings"
    if name.startswith("blocks."):
        return "backbone: attention" if ".attn." in name else "backbone: feed-forward"
    return {
        "cond_embed.weight": "projections",
        "move_embed.weight": "move embedding",
        "ply_embed.weight": "ply embedding",
        "head.weight": "policy head",
    }.get(name, "other")


def weight_matrices(model: nn.Module, only: str | None = None):
    """Yield (name, parameter) for every compressible weight, optionally one group."""
    for module_name, module in model.named_modules():
        if isinstance(module, COMPRESSIBLE):
            name = f"{module_name}.weight" if module_name else "weight"
            if only is None or group_of(name) == only:
                yield name, module.weight


def parameter_groups(model: nn.Module) -> dict[str, int]:
    """Parameters per group, covering every weight matrix exactly once."""
    sizes: dict[str, int] = {}
    for name, weight in weight_matrices(model):
        label = group_of(name)
        sizes[label] = sizes.get(label, 0) + weight.numel()
    return sizes


# ── Methods ───────────────────────────────────────────────────────────────


@torch.no_grad()
def quantize_(weight: torch.Tensor, bits: int) -> None:
    """Symmetric round-to-nearest, one scale per output row, in place."""
    if bits >= 16:
        return
    top = 2 ** (bits - 1) - 1
    scale = (weight.abs().amax(dim=1, keepdim=True) / top).clamp(min=1e-12)
    weight.copy_((weight / scale).round().clamp(-top - 1, top) * scale)


@torch.no_grad()
def prune_(
    weights: list[torch.Tensor], fraction: float, random: bool = False, seed: int = 0
) -> int:
    """Zero the smallest-magnitude fraction of all given weights together.

    One global threshold rather than one per matrix, so layers that matter
    less lose more. With `random` it zeroes the same number of weights chosen
    blindly, the control that shows whether magnitude carried any information.
    Returns how many weights were zeroed.
    """
    flat = torch.cat([w.abs().flatten() for w in weights])
    count = int(fraction * flat.numel())
    if count == 0:
        return 0

    if random:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(flat.numel(), generator=generator).to(flat.device)
        chosen = torch.zeros_like(flat, dtype=torch.bool)
        chosen[order[:count]] = True
    else:
        chosen = torch.zeros_like(flat, dtype=torch.bool)
        chosen[flat.topk(count, largest=False).indices] = True

    start = 0
    for w in weights:
        part = chosen[start : start + w.numel()].view_as(w)
        w.masked_fill_(part, 0.0)
        start += w.numel()
    return count


def low_rank_params(shape: tuple[int, int], rank: int) -> int:
    """What a rank-r factorisation of a matrix actually stores."""
    return rank * (shape[0] + shape[1])


def rank_for(shape: tuple[int, int], keep: float) -> int | None:
    """The rank to keep `keep` of a matrix's singular values, or None if pointless.

    Two factors of rank r cost r(m+n) numbers against m*n for the matrix, so for
    a matrix that is not very rectangular a factorisation only saves anything
    below a rank of m*n/(m+n). Above that it is bigger than the original and no
    competent method would apply it, so the matrix is left alone.
    """
    rank = max(1, round(keep * min(shape)))
    return rank if low_rank_params(shape, rank) < shape[0] * shape[1] else None


@torch.no_grad()
def low_rank_(weight: torch.Tensor, keep: float) -> int | None:
    """Replace a matrix by its best rank-r approximation where that saves bytes.

    Returns the rank used, or None if the matrix was left untouched.
    """
    rank = rank_for(tuple(weight.shape), keep)
    if rank is None:
        return None
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    weight.copy_((u[:, :rank] * s[:rank]) @ vh[:rank])
    return rank
