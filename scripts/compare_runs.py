"""
Put finished runs side by side and apply the decision rule.

    python scripts/compare_runs.py runs/arms/*/ --baseline runs/arms/baseline

The rule was fixed before any arm was run: keep a change if it improves
mate-in-one or the held-out puzzles by at least 2 points without costing more
than 0.5 points of top-1. Two baselines with different seeds measure how far a
result moves by chance, which is printed so the 2-point threshold can be
judged against it.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessdt.constants import ModelConfig  # noqa: E402

#: The rule, in points of accuracy.
MIN_GAIN = 2.0
MAX_TOP1_LOSS = 0.5


def load(run: Path) -> dict | None:
    """The numbers of one run, or None if it has not finished."""
    needed = ("config.json", "results.json", "tactics.json", "evaluation.json")
    if not all((run / name).exists() for name in needed):
        return None
    config, results, tactics, evaluation = (json.loads((run / n).read_text()) for n in needed)
    model = ModelConfig(**config["model"])
    steering = [r["solved"] for r in evaluation.get("steering", {}).values()]
    return {
        "name": run.name,
        "params": results["parameters"] / 1e6,
        "tokens": model.stream_tokens,
        "minutes": results["hours"] * 60,
        "top1": results["val"]["top1"] * 100,
        "top3": results["val"]["top3"] * 100,
        "puzzles": evaluation["puzzle_set"]["solved"] * 100,
        "mate": tactics["model_plays_mate"] * 100,
        "mate_alone": tactics["model_plays_mate_without_history"] * 100,
        "steer": (max(steering) - min(steering)) * 100 if steering else float("nan"),
        "board": evaluation["ablations"]["board"]["changed"] * 100,
    }


def verdict(row: dict, base: dict) -> str:
    """Apply the rule to one arm against the baseline."""
    if row["name"] == base["name"]:
        return "baseline"
    gain_mate, gain_puzzles = row["mate"] - base["mate"], row["puzzles"] - base["puzzles"]
    cost = base["top1"] - row["top1"]
    helps = gain_mate >= MIN_GAIN or gain_puzzles >= MIN_GAIN
    if helps and cost <= MAX_TOP1_LOSS:
        return "KEEP"
    if helps:
        return f"helps but costs {cost:.1f} top-1"
    return "no"


def table(rows: list[dict], base: dict) -> str:
    head = (
        f"{'arm':<16}{'params':>7}{'tok':>6}{'min':>6}{'top-1':>7}{'top-3':>7}{'puzzle':>8}"
        f"{'mate':>7}{'no hist':>8}{'steer':>7}{'board':>7}   verdict"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['name']:<16}{r['params']:>6.1f}M{r['tokens']:>6}{r['minutes']:>6.0f}"
            f"{r['top1']:>7.2f}{r['top3']:>7.2f}{r['puzzles']:>8.2f}{r['mate']:>7.2f}"
            f"{r['mate_alone']:>8.2f}{r['steer']:>7.2f}{r['board']:>7.1f}   {verdict(r, base)}"
        )
    return "\n".join(lines)


def noise_note(rows: list[dict]) -> str | None:
    by_name = {r["name"]: r for r in rows}
    if "baseline" not in by_name or "baseline_seed1" not in by_name:
        return None
    a, b = by_name["baseline"], by_name["baseline_seed1"]
    parts = [f"{k} {abs(a[k] - b[k]):.2f}" for k in ("top1", "puzzles", "mate")]
    gaps = [abs(a["puzzles"] - b["puzzles"]), abs(a["mate"] - b["mate"])]
    note = f"Run-to-run noise (baseline against itself, other seed), in points: {', '.join(parts)}."
    if max(gaps) > MIN_GAIN / 2:
        note += (
            f"\n  WARNING: that is over half the {MIN_GAIN:g}-point threshold, "
            "so small gaps are not trustworthy."
        )
    return note


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--baseline", type=Path, required=True)
    args = ap.parse_args()

    rows = [r for run in args.runs if (r := load(run))]
    unfinished = [run.name for run in args.runs if run.is_dir() and load(run) is None]
    base = next((r for r in rows if r["name"] == args.baseline.name), None)
    if base is None:
        raise SystemExit(f"baseline {args.baseline} has not finished")

    print(table(sorted(rows, key=lambda r: (r["name"] != base["name"], r["name"])), base))
    if note := noise_note(rows):
        print("\n" + note)
    if unfinished:
        print(f"\nNot finished: {', '.join(unfinished)}")
    print(
        f"\nRule: keep if mate or puzzles gain >= {MIN_GAIN:g} points "
        f"and top-1 loses <= {MAX_TOP1_LOSS:g}."
    )


if __name__ == "__main__":
    main()
