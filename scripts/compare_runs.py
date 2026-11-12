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

#: Tactics beyond mate in one: the probe sets and the puzzle themes that go with them.
EXTRA_PROBES = ("mate_in_two", "free_piece", "fork")
EXTRA_THEMES = ("mateIn2", "hangingPiece", "fork")

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
    themes = evaluation["puzzle_set"].get("by_theme", {})
    probes = {}
    for name in EXTRA_PROBES:
        path = run / f"tactics_{name}.json"
        if path.exists():
            probes[name] = json.loads(path.read_text())
    return {
        "themes": {t: themes[t]["solved"] * 100 for t in EXTRA_THEMES if t in themes},
        "probes": probes,
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


def extra_table(rows: list[dict]) -> str | None:
    """Tactics beyond mate in one: puzzle themes, and the model against the humans.

    The probe columns are how often the model's first choice is a move that
    carries out the tactic, on held-out game positions where one exists. The
    humans' own rate is the same for every run, so it is printed once.
    """
    if not any(r["themes"] or r["probes"] for r in rows):
        return None
    head = f"{'arm':<16}" + "".join(f"{'puz ' + t:>17}" for t in EXTRA_THEMES)
    head += "".join(f"{'game ' + p:>18}" for p in EXTRA_PROBES)
    lines = [head, "-" * len(head)]
    for r in rows:
        cells = "".join(
            f"{r['themes'][t]:>17.2f}" if t in r["themes"] else f"{'-':>17}" for t in EXTRA_THEMES
        )
        cells += "".join(
            f"{r['probes'][p]['model_plays_mate'] * 100:>18.2f}"
            if p in r["probes"]
            else f"{'-':>18}"
            for p in EXTRA_PROBES
        )
        lines.append(f"{r['name']:<16}" + cells)
    humans = {
        p: next(r["probes"][p]["human_plays_mate"] * 100 for r in rows if p in r["probes"])
        for p in EXTRA_PROBES
        if any(p in r["probes"] for r in rows)
    }
    if humans:
        lines.append(
            "humans: " + ", ".join(f"{p} {v:.1f}%" for p, v in humans.items()) + " (game columns)"
        )
    return "\n".join(lines)


#: Pairs of the same run under two seeds: the gap inside a pair is chance.
SEED_PAIRS = (("baseline", "baseline_seed1"), ("t25", "t25_seed1"))


def noise_note(rows: list[dict]) -> str | None:
    by_name = {r["name"]: r for r in rows}
    notes = []
    for first, second in SEED_PAIRS:
        if first not in by_name or second not in by_name:
            continue
        a, b = by_name[first], by_name[second]
        parts = [f"{k} {abs(a[k] - b[k]):.2f}" for k in ("top1", "puzzles", "mate")]
        gaps = [abs(a["puzzles"] - b["puzzles"]), abs(a["mate"] - b["mate"])]
        note = (
            f"Run-to-run noise ({first} against itself, other seed), in points: {', '.join(parts)}."
        )
        if max(gaps) > MIN_GAIN / 2:
            note += (
                f"\n  WARNING: that is over half the {MIN_GAIN:g}-point threshold, "
                "so small gaps are not trustworthy."
            )
        notes.append(note)
    return "\n".join(notes) or None


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
    if more := extra_table(sorted(rows, key=lambda r: (r["name"] != base["name"], r["name"]))):
        print("\nBeyond mate in one:\n" + more)
    if unfinished:
        print(f"\nNot finished: {', '.join(unfinished)}")
    print(
        f"\nRule: keep if mate or puzzles gain >= {MIN_GAIN:g} points "
        f"and top-1 loses <= {MAX_TOP1_LOSS:g}."
    )


if __name__ == "__main__":
    main()
