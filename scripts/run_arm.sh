#!/usr/bin/env bash
# Train one experimental arm and measure it, start to finish.
#
#   bash scripts/run_arm.sh NAME [train.py options...]
#
# Trains, then runs the mate-in-one probe and the full evaluation (accuracy,
# ablations, puzzles, steering). Everything lands in runs/arms/NAME/, and the
# training log in runs/arms/NAME.log. Settings come from the environment:
#   PY=python  DATA=data/encoded  PUZZLES=data/puzzles  TACTICS=data/tactics_all
#   STEPS=8000  BATCH=256
#   FINAL_WINDOWS=6000  PUZZLE_SAMPLE=20000  EVAL_EVERY=2000   (smaller for a quick check)
#
# Safe to run again after an interruption: each stage is skipped if it already
# finished, and training picks up from its last checkpoint instead of starting over.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
DATA=${DATA:-data/encoded}
PUZZLES=${PUZZLES:-data/puzzles}
TACTICS=${TACTICS:-data/tactics_all}
STEPS=${STEPS:-8000}
BATCH=${BATCH:-256}
FINAL_WINDOWS=${FINAL_WINDOWS:-6000}
PUZZLE_SAMPLE=${PUZZLE_SAMPLE:-20000}
EVAL_EVERY=${EVAL_EVERY:-2000}

name=$1
shift
out=runs/arms/$name
mkdir -p runs/arms

if [ -f "$out/results.json" ]; then
  echo "[$name] training already finished"
else
  if [ -f "$out/resume.pt" ]; then echo "[$name] resuming from its last checkpoint"; else echo "[$name] training: $*"; fi
  # Appended, so an interrupted run keeps the log of what it already did.
  $PY -u scripts/train.py --data "$DATA" --out "$out" --steps "$STEPS" --batch "$BATCH" \
    --eval-every "$EVAL_EVERY" --eval-windows 2000 --final-windows "$FINAL_WINDOWS" "$@" >> "$out.log" 2>&1
fi

if [ -f "$out/tactics.json" ]; then
  echo "[$name] mate probe already done"
else
  echo "[$name] measuring"
  $PY scripts/probe_tactics.py "$out/model_best.pt" --data "$DATA" --set "$TACTICS/mate_in_one.npz" \
    > "$out/tactics.log" 2>&1
fi
if [ ! -f "$out/evaluation.json" ]; then
  $PY scripts/evaluate.py "$out/model_best.pt" --data "$DATA" --puzzles "$PUZZLES" \
    --puzzle-sample "$PUZZLE_SAMPLE" --windows 1500 --ablation-windows 500 > "$out/evaluation.log" 2>&1
fi
echo "[$name] done"
