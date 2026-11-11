"""
Train the reference model.

    python scripts/train.py --data data/encoded --out runs/reference --hours 6

Give it a time budget and it fits the learning-rate schedule to what the
machine can do in that time, so the run ends when you were told it would. Give
it --steps instead for a fixed-length run.

Safe to interrupt: rerun the same command and it resumes from the last
checkpoint, schedule and all.

Writes into the run directory:

    config.json      everything needed to say what this run was
    metrics.jsonl    one line per log interval and per evaluation
    model_best.pt    weights at the best validation accuracy
    model_last.pt    weights at the last evaluation
    results.json     final numbers on the full validation and test sets
"""

import argparse
import ast
import json
import math
import sys
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessdt.constants import PRESETS, SIZE_CAP_BYTES  # noqa: E402
from chessdt.data import Corpus  # noqa: E402
from chessdt.evaluate import autocast, score  # noqa: E402
from chessdt.model import ChessDT, load_checkpoint, save_checkpoint  # noqa: E402
from chessdt.splits import TEST, VAL  # noqa: E402

#: A time budget becomes a step count by timing the steps in this range. The
#: first few are left out because they carry one-off start-up costs.
CALIBRATION_FROM, CALIBRATION_STEPS = 20, 100


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def learning_rate(step: int, total: int, peak: float, warmup: int) -> float:
    """Linear warmup, then cosine decay to 5% of the peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    return peak * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def build_optimizer(model: ChessDT, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW with decay on the weight matrices only."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": rest, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def log(run: Path, record: dict) -> None:
    with (run / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/encoded"))
    ap.add_argument("--out", type=Path, default=Path("runs/reference"))
    ap.add_argument("--hours", type=float, default=6.0, help="time budget for the run")
    ap.add_argument("--steps", type=int, default=None, help="fixed length; overrides --hours")
    ap.add_argument("--batch", type=int, default=256, help="windows per step")
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-windows", type=int, default=3000)
    ap.add_argument(
        "--final-windows",
        type=int,
        default=None,
        help="cap the end-of-run scoring; for smoke runs only",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="v1", help="model size")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="OPTION=VALUE",
        help="override a model option, e.g. --set summary_tokens=4 --set geometry=True",
    )
    ap.add_argument(
        "--tactics", type=Path, default=None, help="mate-in-one set, with the train split"
    )
    ap.add_argument(
        "--tactical-fraction",
        type=float,
        default=0.0,
        help="share of windows built around a mate-in-one position",
    )
    ap.add_argument(
        "--micro",
        type=int,
        default=1,
        help="run each batch in this many pieces, for models that do not fit at once",
    )
    ap.add_argument("--device", default=pick_device())
    args = ap.parse_args()
    if args.eval_every < CALIBRATION_STEPS:
        ap.error(f"--eval-every must be at least {CALIBRATION_STEPS}")

    run, device = args.out, args.device
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    known = {f.name for f in fields(PRESETS[args.preset])}
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        if key not in known:
            ap.error(f"--set {key}: not a model option; choose from {sorted(known)}")
        overrides[key] = ast.literal_eval(value)
    cfg = replace(PRESETS[args.preset], **overrides)
    if cfg.artifact_bytes() > SIZE_CAP_BYTES:
        ap.error(f"this model is {cfg.artifact_bytes() / 1024**2:.1f} MiB, over the 100 MiB cap")
    if args.tactical_fraction and not args.tactics:
        ap.error("--tactical-fraction needs --tactics")
    plies = cfg.context_plies
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else ""))

    corpus = Corpus(args.data, device)
    val_starts = corpus.split_starts(int(VAL), plies, limit=args.eval_windows)
    print(f"Corpus: {len(corpus):,} positions, {len(corpus.train_positions):,} for training")
    if args.tactical_fraction:
        found = corpus.load_tactics(args.tactics)
        print(f"Tactics: {found:,} training positions with a mate in one, "
              f"{args.tactical_fraction:.0%} of windows built around one")  # fmt: skip

    model = ChessDT(cfg).to(device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device == "cuda" and not torch.cuda.is_bf16_supported()
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    print(
        f"Model: {sum(p.numel() for p in model.parameters()):,} parameters "
        f"({args.preset}{', ' + ', '.join(args.set) if args.set else ''}), "
        f"{cfg.stream_tokens} tokens per window"
    )

    step, total, best, elapsed = 0, args.steps, 0.0, 0.0
    resume = run / "resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        # Reseeded rather than restored: generator state is not portable
        # across devices, and a fresh stream from here on is all that matters.
        generator.manual_seed(args.seed + state["step"])
        step, total, best, elapsed = state["step"], state["total"], state["best"], state["elapsed"]
        print(f"Resumed at step {step:,} of {total:,} (best val top-1 {best:.4f})")
    else:
        (run / "config.json").write_text(
            json.dumps(
                {
                    "model": asdict(cfg),
                    "args": {k: str(v) for k, v in vars(args).items()},
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name() if device == "cuda" else device,
                },
                indent=2,
            )
        )

    def checkpoint() -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "total": total,
                "best": best,
                "elapsed": elapsed,
            },
            resume,
        )

    seen_loss, seen_hits, seen_count = 0.0, 0, 0
    mark = time.monotonic()
    calibration_mark = None  # its own clock: `mark` is reset at every log line
    model.train()

    while total is None or step < total:
        # With a time budget the run length is fixed once throughput is known.
        # That happens well inside warmup, so the schedule has no seam.
        horizon = total if total is not None else 10**9
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, horizon, args.lr, args.warmup)

        batch = corpus.sample_train(args.batch, plies, generator, args.tactical_fraction)
        wanted = int(batch.valid.sum())
        optimizer.zero_grad(set_to_none=True)
        loss_sum, hits, count = 0.0, 0, 0
        for piece in batch.split(args.micro):
            with autocast(device):
                logits = model(*piece.inputs())
            logits, target = logits.float()[piece.valid], piece.moves[piece.valid]
            piece_loss = F.cross_entropy(logits, target, reduction="sum") / wanted
            scaler.scale(piece_loss).backward()
            loss_sum += piece_loss.item() * wanted
            hits += (logits.argmax(-1) == target).sum().item()
            count += len(target)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1

        seen_loss += loss_sum
        seen_hits += hits
        seen_count += count

        if total is None and step == CALIBRATION_FROM:
            calibration_mark = time.monotonic()
        if total is None and step == CALIBRATION_STEPS:
            timed = CALIBRATION_STEPS - CALIBRATION_FROM
            per_step = (time.monotonic() - calibration_mark) / timed
            total = max(int(args.hours * 3600 / per_step), args.warmup * 2)
            passes = total * args.batch * plies / len(corpus.train_positions)
            print(
                f"Calibrated: {per_step * 1000:.0f} ms/step, so {args.hours:g} h is "
                f"{total:,} steps, {passes:.1f} passes over the data"
            )

        if step % args.log_every == 0:
            now = time.monotonic()
            elapsed += now - mark
            rate = seen_count / (now - mark)
            mark = now
            remaining = (total - step) * args.batch * plies / rate / 60 if total else float("nan")
            record = {
                "step": step,
                "train_loss": round(seen_loss / seen_count, 4),
                "train_top1": round(seen_hits / seen_count, 4),
                "lr": round(optimizer.param_groups[0]["lr"], 7),
                "positions_per_s": round(rate),
                "elapsed_min": round(elapsed / 60, 1),
            }
            log(run, record)
            print(
                f"step {step:>7,}  loss {record['train_loss']:.4f}  "
                f"top1 {record['train_top1']:.4f}  lr {record['lr']:.2e}  "
                f"{rate:>7,.0f} pos/s  {remaining:>5.0f} min left",
                flush=True,
            )
            seen_loss, seen_hits, seen_count = 0.0, 0, 0

        if step % args.eval_every == 0 or step == total:
            result = score(model, corpus, val_starts, plies, int(VAL))
            model.train()
            improved = result["top1"] > best
            best = max(best, result["top1"])
            log(
                run,
                {
                    "step": step,
                    "val_loss": result["loss"],
                    "val_top1": result["top1"],
                    "val_top3": result["top3"],
                },
            )
            print(
                f"          val loss {result['loss']:.4f}  val top1 {result['top1']:.4f}"
                f"  top3 {result['top3']:.4f}" + ("  (best so far)" if improved else ""),
                flush=True,
            )
            save_checkpoint(run / "model_last.pt", model, step=step, val_top1=result["top1"])
            if improved:
                save_checkpoint(run / "model_best.pt", model, step=step, val_top1=result["top1"])
            checkpoint()
            mark = time.monotonic()

    print("\nTraining finished. Scoring the best checkpoint on all held-out games ...")
    model, payload = load_checkpoint(run / "model_best.pt", device)

    def final_starts(split: int) -> torch.Tensor:
        return corpus.split_starts(split, plies, limit=args.final_windows)

    results = {
        "best_step": payload["step"],
        "steps": step,
        "hours": round(elapsed / 3600, 2),
        "parameters": sum(p.numel() for p in model.parameters()),
        "val": score(model, corpus, final_starts(int(VAL)), plies, int(VAL)),
        "test": score(model, corpus, final_starts(int(TEST)), plies, int(TEST)),
    }
    (run / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n  best checkpoint   step {results['best_step']:,} of {step:,}")
    for name in ("val", "test"):
        r = results[name]
        print(
            f"  {name:<5} top-1 {r['top1'] * 100:6.2f}%   top-3 {r['top3'] * 100:6.2f}%   "
            f"loss {r['loss']:.4f}   ({r['positions']:,} positions)"
        )
    print(f"\nResults in {run / 'results.json'}")


if __name__ == "__main__":
    main()
