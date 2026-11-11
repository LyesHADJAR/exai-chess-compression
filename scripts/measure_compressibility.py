"""
Gate 3: do standard compression methods separate on this model?

    python scripts/measure_compressibility.py runs/reference/model_best.pt \
        --data data/encoded --puzzles data/puzzles

Applies the usual data-free methods to the released weights, at several
strengths, and measures what each costs. The point is to check, before the
challenge is announced, that the scoring axis ranks submissions: a gentle
setting must be nearly free, a harsh one must break the model, and different
methods must land at visibly different places in between. If everything ties,
nobody can be ranked on skill.

Fidelity is measured two ways, both against the *uncompressed* model:

  agreement  the share of held-out positions where the compressed model's first
             choice is the reference model's first choice. Needs no labels,
             which is why it is the agreement set's job.
  puzzles    the solve rate on a fixed sample, with illegal moves masked.

It also reports sensitivity by parameter group: quantise one group to 4 bits,
leave the rest alone. That shows where the model is fragile, which is what a
good submission exploits and a blanket one pays for.

Writes `compressibility.json` beside the checkpoint.
"""

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessdt.compression import (  # noqa: E402
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
from chessdt.data import Corpus  # noqa: E402
from chessdt.evaluate import PuzzleSet, predict  # noqa: E402
from chessdt.model import load_checkpoint  # noqa: E402
from chessdt.splits import TEST  # noqa: E402


def release(device: str) -> None:
    """Hand back the memory a finished variant held.

    Every variant is a full copy of the model. Without this the allocator keeps
    them all cached, and on a machine short of memory the sweep slows from a
    minute per variant to several as the copies pile up.
    """
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


class Probe:
    """A fixed evaluation set, with the reference model's answers cached.

    The reference is run once. Every compressed variant is then compared with
    those stored answers, so the cost of a variant is one forward pass.
    """

    def __init__(self, model, corpus: Corpus, windows: int, puzzles: PuzzleSet | None):
        self.corpus, self.puzzles = corpus, puzzles
        plies = model.cfg.context_plies
        starts = corpus.split_starts(int(TEST), plies, limit=windows)
        self.batches = []
        for i in range(0, len(starts), 512):
            batch = corpus.window(starts[i : i + 512], plies)
            self.batches.append((batch, batch.valid & (batch.labels == int(TEST))))

        self.reference = self._logprobs(model)
        self.reference_puzzles = puzzles.solve(model) if puzzles else None
        self.positions = sum(int(keep.sum()) for _, keep in self.batches)

    def _logprobs(self, model) -> list[torch.Tensor]:
        model.eval()
        return [
            F.log_softmax(predict(model, batch, self.corpus.device)[keep], dim=-1).cpu()
            for batch, keep in self.batches
        ]

    def measure(self, model) -> dict:
        agree, kl, human, total = 0, 0.0, 0, 0
        for (batch, keep), ref in zip(self.batches, self.reference, strict=True):
            now = F.log_softmax(predict(model, batch, self.corpus.device)[keep], dim=-1).cpu()
            if not torch.isfinite(now).all():
                return {"agreement": 0.0, "kl": float("inf"), "top1": 0.0, "puzzles": None}
            agree += (now.argmax(-1) == ref.argmax(-1)).sum().item()
            kl += F.kl_div(now, ref, log_target=True, reduction="sum").item()
            human += (now.argmax(-1).to(batch.moves.device) == batch.moves[keep]).sum().item()
            total += len(ref)

        result = {"agreement": agree / total, "kl": kl / total, "top1": human / total}
        result["puzzles"] = self.puzzles.solve(model)["solved"] if self.puzzles else None
        return result


def size_mb(model, bits: dict[str, float], kept: dict[str, float] | None = None) -> float:
    """Stored size if each group is held at the given bits per weight.

    Everything that is not a weight matrix stays at 32 bits. A quantised group
    also pays for one 16-bit scale per output row, which a real file has to
    carry. `kept` is the share of weights still stored, for pruning (with no
    cost for the mask, which is optimistic) and for low-rank factorisation.
    """
    matrices = {name: w.shape for name, w in weight_matrices(model)}
    others = sum(p.numel() for p in model.parameters()) - sum(r * c for r, c in matrices.values())

    total = others * 32.0
    for name, (rows, cols) in matrices.items():
        group = group_of(name)
        width = bits.get(group, 32.0)
        total += rows * cols * (kept or {}).get(group, 1.0) * width
        if width < 16:
            total += rows * 16
    return total / 8 / 1024**2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/encoded"))
    ap.add_argument("--puzzles", type=Path, default=None)
    ap.add_argument("--windows", type=int, default=1500, help="test windows of 16 plies")
    ap.add_argument("--puzzle-sample", type=int, default=5000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    reference, _ = load_checkpoint(args.checkpoint, device)
    master = copy.deepcopy(reference).cpu()  # variants start from this, on the CPU
    corpus = Corpus(args.data, device)
    puzzles = PuzzleSet.load(args.puzzles, args.puzzle_sample, device) if args.puzzles else None

    print("Caching the reference model's answers ...", flush=True)
    probe = Probe(reference, corpus, args.windows, puzzles)
    base = probe.measure(reference)
    full_mb = size_mb(reference, {})
    print(
        f"{probe.positions:,} held-out positions"
        + (f", {len(puzzles):,} puzzles" if puzzles else "")
        + f". Reference: {full_mb:.2f} MB, human top-1 {base['top1'] * 100:.2f}%"
        + (f", puzzles {base['puzzles'] * 100:.2f}%" if puzzles else "")
        + "\n"
    )

    rows: list[dict] = []
    sensitivity: dict[str, dict] = {}
    started = time.monotonic()

    out = args.checkpoint.parent / "compressibility.json"

    def save() -> None:
        """Write what is known so far, so an interrupted sweep loses nothing."""
        out.write_text(
            json.dumps(
                {
                    "reference": {"mb": full_mb, **base},
                    "variants": rows,
                    "sensitivity": sensitivity,
                },
                indent=2,
            )
        )

    def run(family: str, setting: str, mb: float, apply) -> None:
        began = time.monotonic()
        model = copy.deepcopy(master)
        apply(model)
        model.to(device)
        r = probe.measure(model)
        del model
        release(device)
        row = {"family": family, "setting": setting, "mb": mb, "ratio": full_mb / mb, **r}
        rows.append(row)
        save()
        pz = f"{r['puzzles'] * 100:6.2f}%" if r["puzzles"] is not None else "     -"
        print(
            f"  {setting:<26}{mb:>8.2f} MB {full_mb / mb:>6.1f}x"
            f"   agree {r['agreement'] * 100:6.2f}%   puzzles {pz}   kl {r['kl']:.4f}"
            f"   ({time.monotonic() - began:.0f}s)",
            flush=True,
        )

    def every_weight(model):
        return [w for _, w in weight_matrices(model)]

    print("QUANTISATION  one scale per output row, round to nearest")
    for bits in (8, 6, 5, 4, 3, 2):
        run(
            "quantise",
            f"int{bits}",
            size_mb(reference, dict.fromkeys(GROUPS, bits)),
            lambda m, b=bits: [quantize_(w, b) for w in every_weight(m)],
        )

    # Pruning is only a saving if the zeros are not stored. The size below counts
    # the surviving weights and nothing for the index that says where they are,
    # so it is a best case; a real sparse format costs more.
    print("\nPRUNING  one global threshold, smallest weights first")
    print("  (size is a best case: surviving weights only, no cost for the mask)")

    def kept(frac: float) -> dict[str, float]:
        return dict.fromkeys(GROUPS, 1 - frac)

    for frac in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        run(
            "prune",
            f"magnitude {frac:.0%}",
            size_mb(reference, {}, kept(frac)),
            lambda m, f=frac: prune_(every_weight(m), f),
        )
    for frac in (0.3, 0.5):
        run(
            "prune",
            f"random {frac:.0%} (control)",
            size_mb(reference, {}, kept(frac)),
            lambda m, f=frac: prune_(every_weight(m), f, random=True),
        )

    print("\nLOW RANK  truncated SVD, only where the two factors are smaller than the matrix")
    for keep in (0.5, 0.35, 0.25, 0.15, 0.1):

        def low_rank(m, k=keep):
            for _, w in weight_matrices(m):
                low_rank_(w, k)

        saved = 0
        for _, w in weight_matrices(reference):
            rank = rank_for(tuple(w.shape), keep)
            saved += w.numel() - low_rank_params(tuple(w.shape), rank) if rank else 0
        weights = sum(w.numel() for _, w in weight_matrices(reference))
        mb = size_mb(reference, {}, dict.fromkeys(GROUPS, 1 - saved / weights))
        run("low rank", f"rank {keep:.0%}", mb, low_rank)

    print("\nSENSITIVITY  one group at int4, everything else untouched")
    sizes = parameter_groups(reference)
    print(f"  {'group':<30}{'weights':>11}{'share':>8}   agree   puzzles")
    for group in GROUPS:
        model = copy.deepcopy(master)
        for _, w in weight_matrices(model, only=group):
            quantize_(w, 4)
        model.to(device)
        r = probe.measure(model)
        del model
        release(device)
        sensitivity[group] = {"weights": sizes[group], **r}
        save()
        share = sizes[group] / sum(sizes.values())
        pz = f"{r['puzzles'] * 100:6.2f}%" if r["puzzles"] is not None else "     -"
        print(
            f"  {group:<30}{sizes[group]:>11,}{share:>7.1%}   {r['agreement'] * 100:5.1f}%   {pz}"
        )

    print(f"\nWrote {out}  ({(time.monotonic() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
