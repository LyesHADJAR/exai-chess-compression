"""The comparison table and the decision rule it applies."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from chessdt.constants import ModelConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_runs import load, noise_note, verdict  # noqa: E402


def make_run(root, name, *, top1=40.0, puzzles=30.0, mate=30.0, **options):
    run = root / name
    run.mkdir()
    cfg = ModelConfig(**options)
    (run / "config.json").write_text(json.dumps({"model": asdict(cfg), "args": {}}))
    (run / "results.json").write_text(json.dumps({
        "parameters": cfg.parameter_count(), "hours": 0.5,
        "val": {"top1": top1 / 100, "top3": 0.7},
    }))  # fmt: skip
    (run / "tactics.json").write_text(json.dumps({
        "model_plays_mate": mate / 100, "model_plays_mate_without_history": 0.2,
    }))  # fmt: skip
    (run / "evaluation.json").write_text(json.dumps({
        "puzzle_set": {"solved": puzzles / 100},
        "ablations": {"board": {"changed": 0.99}},
        "steering": {"a": {"solved": 0.30}, "b": {"solved": 0.32}},
    }))  # fmt: skip
    return run


def rows(root, **arms):
    base = load(make_run(root, "baseline"))
    return base, {name: load(make_run(root, name, **kw)) for name, kw in arms.items()}


def test_a_run_missing_a_measurement_is_not_loaded(tmp_path):
    run = make_run(tmp_path, "x")
    (run / "tactics.json").unlink()
    assert load(run) is None


def test_the_stream_length_comes_from_the_config(tmp_path):
    assert load(make_run(tmp_path, "a", summary_tokens=4))["tokens"] == 16 * 6


def test_an_arm_that_gains_enough_without_cost_is_kept(tmp_path):
    base, arms = rows(tmp_path, a=dict(mate=32.5))
    assert verdict(arms["a"], base) == "KEEP"


def test_a_gain_in_either_measure_is_enough(tmp_path):
    base, arms = rows(tmp_path, a=dict(puzzles=32.0))
    assert verdict(arms["a"], base) == "KEEP"


def test_a_gain_below_the_threshold_is_not_enough(tmp_path):
    base, arms = rows(tmp_path, a=dict(mate=31.9, puzzles=31.9))
    assert verdict(arms["a"], base) == "no"


def test_a_gain_that_costs_too_much_top1_is_not_kept(tmp_path):
    base, arms = rows(tmp_path, a=dict(mate=35.0, top1=39.0))
    assert "costs 1.0" in verdict(arms["a"], base)


def test_a_small_top1_loss_is_tolerated(tmp_path):
    base, arms = rows(tmp_path, a=dict(mate=35.0, top1=39.6))
    assert verdict(arms["a"], base) == "KEEP"


def test_the_baseline_is_labelled_as_such(tmp_path):
    base, _ = rows(tmp_path)
    assert verdict(base, base) == "baseline"


def test_noise_is_reported_and_flagged_when_it_swamps_the_threshold(tmp_path):
    make_run(tmp_path, "baseline", mate=30.0, puzzles=30.0)
    make_run(tmp_path, "baseline_seed1", mate=31.5, puzzles=30.2)
    note = noise_note([load(tmp_path / "baseline"), load(tmp_path / "baseline_seed1")])
    assert "mate 1.50" in note and "WARNING" in note


def test_no_noise_note_without_both_baselines(tmp_path):
    assert noise_note([load(make_run(tmp_path, "baseline"))]) is None


@pytest.mark.parametrize("gain", [2.0, 2.5, 10.0])
def test_the_threshold_is_inclusive(tmp_path, gain):
    base, arms = rows(tmp_path, a=dict(mate=30.0 + gain))
    assert verdict(arms["a"], base) == "KEEP"


def test_a_run_without_the_extra_probes_still_loads(tmp_path):
    run = load(make_run(tmp_path, "a"))
    assert run["probes"] == {} and run["themes"] == {}


def test_extra_probes_and_themes_are_read_when_present(tmp_path):
    from compare_runs import extra_table

    run = make_run(tmp_path, "a")
    (run / "tactics_fork.json").write_text(
        json.dumps({"human_plays_mate": 0.3, "model_plays_mate": 0.25})
    )
    evaluation = json.loads((run / "evaluation.json").read_text())
    evaluation["puzzle_set"]["by_theme"] = {"fork": {"solved": 0.4, "puzzles": 10}}
    (run / "evaluation.json").write_text(json.dumps(evaluation))
    table = extra_table([load(run)])
    assert "25.00" in table and "40.00" in table and "humans: fork 30.0%" in table


def test_no_extra_table_when_nothing_extra_was_measured(tmp_path):
    from compare_runs import extra_table

    assert extra_table([load(make_run(tmp_path, "a"))]) is None


def test_the_t25_seed_pair_reports_its_own_noise(tmp_path):
    make_run(tmp_path, "t25", mate=30.0, puzzles=30.0)
    make_run(tmp_path, "t25_seed1", mate=30.4, puzzles=30.4)
    note = noise_note([load(tmp_path / "t25"), load(tmp_path / "t25_seed1")])
    assert "t25 against itself" in note and "WARNING" not in note
