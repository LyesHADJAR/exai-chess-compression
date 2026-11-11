#!/usr/bin/env bash
# Run a set of experimental arms, two at a time, then print the comparison.
#
#   bash scripts/run_wave.sh core      # the cheap, most informative arms
#   bash scripts/run_wave.sh extra     # the larger ones
#   bash scripts/run_wave.sh all
#
# Arms share the GPU, so two at once is mostly about keeping it busy. Add
# `PARALLEL=1` for one at a time. Rerunning skips any arm that already finished.
set -uo pipefail
cd "$(dirname "$0")/.."

export PY=${PY:-python}
PARALLEL=${PARALLEL:-2}
group=${1:-core}

# name|options. The baseline appears twice, with different seeds: the gap
# between those two is how much a result can move by chance, which is what
# decides whether any other gap means anything.
core=(
  "baseline|"
  "baseline_seed1|--seed 1"
  "pooled8|--set summary_tokens=8"
  "geometry|--set geometry=True"
  "tactical|--tactics ${TACTICS:-data/tactics_all}/mate_in_one.npz --tactical-fraction 0.5"
  "conddrop|--set cond_dropout=0.1"
)
extra=(
  "pooled4|--set summary_tokens=4"
  "m18|--preset m18"
  "m25|--preset m25"
  "fulldetail|--set full_detail=True --set enc_layers=1 --set layers=6 --micro 4"
)

case "$group" in
  core) arms=("${core[@]}") ;;
  extra) arms=("${extra[@]}") ;;
  all) arms=("${core[@]}" "${extra[@]}") ;;
  *) echo "usage: $0 core|extra|all"; exit 2 ;;
esac

run() {
  local name=${1%%|*} options=${1#*|}
  if [ -f "runs/arms/$name/evaluation.json" ]; then echo "[$name] already done"; return; fi
  rm -rf "runs/arms/$name"
  # shellcheck disable=SC2086
  bash scripts/run_arm.sh "$name" $options || echo "[$name] FAILED, see runs/arms/$name.log"
}
export -f run

printf '%s\n' "${arms[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'run "$1"' _ {}
echo
$PY scripts/compare_runs.py runs/arms/*/ --baseline runs/arms/baseline
