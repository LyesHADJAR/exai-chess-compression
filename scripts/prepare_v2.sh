#!/usr/bin/env bash
# Everything the second round of arms needs beyond the corpus itself.
#
#   bash scripts/prepare_v2.sh
#
# Scans the v2 corpus for mate-in-one positions in every split (the training
# ones drive tactical emphasis, the held-out ones are what v2 is judged on) and
# rebuilds the puzzle set against the v2 corpus, so no puzzle shares a position
# with anything the model trains on. Then the arms run with:
#
#   DATA=data/v2 TACTICS=data/v2_tactics PUZZLES=data/puzzles_v2 \
#     bash scripts/run_wave.sh <group>
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
CORPUS=${CORPUS:-data/v2}

[ -f "$CORPUS/split_labels.npy" ] || { echo "$CORPUS is not built yet; run build_v2_corpus.sh"; exit 1; }

$PY scripts/build_tactics_set.py --corpus "$CORPUS" --splits train val test --out "${CORPUS}_tactics"
$PY scripts/build_puzzles.py --corpus "$CORPUS" --out data/puzzles_v2
echo "Ready. Run the arms with DATA=$CORPUS TACTICS=${CORPUS}_tactics PUZZLES=data/puzzles_v2"
