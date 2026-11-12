#!/usr/bin/env bash
# Run a set of experimental arms, two at a time, then print the comparison.
#
#   bash scripts/run_wave.sh core      # the cheap, most informative arms
#   bash scripts/run_wave.sh extra     # the larger ones
#   bash scripts/run_wave.sh round2   # tactical emphasis, tuned, and two sizes
#   bash scripts/run_wave.sh round3   # tactics beyond mate in one
#   bash scripts/run_wave.sh all
#
# Arms share the GPU, so two at once is mostly about keeping it busy. Add
# `PARALLEL=1` for one at a time. Rerunning skips any arm that already finished
# and resumes any that were interrupted from their last checkpoint, so it is the
# right thing to run again after a restart. To start an arm over, delete its
# folder in runs/arms first.
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

# Second round. Tactical emphasis was the one change that helped, at a cost of
# top-1, so these map how much is needed and whether fading it out recovers the
# cost. The two size arms ask whether a bigger model gets more from the same help.
T=${TACTICS:-data/tactics_all}/mate_in_one.npz
round2=(
  "t10|--tactics $T --tactical-fraction 0.1"
  "t25|--tactics $T --tactical-fraction 0.25"
  "t40|--tactics $T --tactical-fraction 0.4"
  "t50_anneal|--tactics $T --tactical-fraction 0.5 --tactical-anneal 0.7"
  "m18_t25|--preset m18 --tactics $T --tactical-fraction 0.25"
  "m25_t25|--preset m25 --tactics $T --tactical-fraction 0.25"
)

# Third round. Mate-in-one emphasis (t25 above) is the recipe to beat. These add
# a forced mate in two, free pieces and forks to the positions the training windows
# are built around, and ask whether that lifts the puzzles it should. t25_seed1
# repeats the recipe with another seed, to say how far a result moves by chance.
TD=${TACTICS:-data/tactics_all}
M1=$TD/mate_in_one.npz M2=$TD/mate_in_two.npz FP=$TD/free_piece.npz FK=$TD/fork.npz
round3=(
  "t25_seed1|--seed 1 --tactics $M1 --tactical-fraction 0.25"
  "x25_mates|--tactics $M1 $M2 --tactical-fraction 0.25"
  "x25_all|--tactics $M1 $M2 $FP $FK --tactical-fraction 0.25"
  "x25_quiet|--tactics $FP $FK --tactical-fraction 0.25"
  "x40_all|--tactics $M1 $M2 $FP $FK --tactical-fraction 0.4"
)

case "$group" in
  round3) arms=("${round3[@]}") ;;
  round2) arms=("${round2[@]}") ;;
  core) arms=("${core[@]}") ;;
  extra) arms=("${extra[@]}") ;;
  all) arms=("${core[@]}" "${extra[@]}") ;;
  *) echo "usage: $0 core|extra|round2|round3|all"; exit 2 ;;
esac

# t25 was measured before the new probes and the theme breakdown existed. Training is
# not repeated: only its measurements are redone, so the new columns have a baseline.
# The old evaluation is kept as evaluation_round2.json.
if [ "$group" = round3 ] && [ -f runs/arms/t25/results.json ] \
    && [ ! -f runs/arms/t25/tactics_fork.json ]; then
  cp runs/arms/t25/evaluation.json runs/arms/t25/evaluation_round2.json
  rm -f runs/arms/t25/evaluation.json
  bash scripts/run_arm.sh t25 --tactics "$M1" --tactical-fraction 0.25
fi

run() {
  local name=${1%%|*} options=${1#*|}
  if [ -f "runs/arms/$name/evaluation.json" ]; then echo "[$name] already done"; return; fi
  # shellcheck disable=SC2086
  bash scripts/run_arm.sh "$name" $options || echo "[$name] FAILED, see runs/arms/$name.log"
}
export -f run

printf '%s\n' "${arms[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'run "$1"' _ {}
echo
# Round three is judged against the recipe it tries to improve, not the bare baseline.
base=baseline
[ "$group" = round3 ] && base=t25
$PY scripts/compare_runs.py runs/arms/*/ --baseline "runs/arms/$base"
