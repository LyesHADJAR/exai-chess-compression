#!/usr/bin/env bash
# Build the v2 training corpus.
#
# Extracts several months at a low rating floor in parallel, merges them with the
# reference corpus, and builds the splits. The months are independent dumps, so
# there is nothing to coordinate between the processes and no game appears twice.
#
#   bash scripts/build_v2_corpus.sh
#
# Settings can be overridden from the environment:
#   MONTHS="2026-08 2026-07"  GAMES=350000  MIN_ELO=1000  PREFETCH=0.4
#   BASE=data/encoded  PARTS=data/v2_parts  OUT=data/v2  PY=python
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
MONTHS=${MONTHS:-"2026-08 2026-07 2026-06 2026-05"}
GAMES=${GAMES:-350000}
MIN_ELO=${MIN_ELO:-1000}
PREFETCH=${PREFETCH:-0.4}
BASE=${BASE:-data/encoded}
PARTS=${PARTS:-data/v2_parts}
OUT=${OUT:-data/v2}
export PYTHONPATH="${PYTHONPATH:-}:$PWD/src"

echo "Free disk here: $(df -h . | awk 'NR==2 {print $4}')"
mkdir -p "$PARTS"

pids=()
for month in $MONTHS; do
  if [ -f "$PARTS/$month/games.npz" ]; then
    echo "$month: already extracted, skipping"
    continue
  fi
  rm -rf "${PARTS:?}/$month"
  $PY -u scripts/extract.py --month "$month" --min-elo "$MIN_ELO" --games "$GAMES" \
    --prefetch-gb "$PREFETCH" --report-every 50000 --out "$PARTS/$month" \
    > "$PARTS/$month.log" 2>&1 &
  pids+=($!)
  echo "$month: started (pid $!), log $PARTS/$month.log"
done

# A background failure does not trip `set -e`, so wait for each job and check it.
failed=0
for pid in "${pids[@]:-}"; do
  [ -n "$pid" ] && { wait "$pid" || failed=1; }
done
if [ "$failed" -ne 0 ]; then
  echo "At least one extraction failed. Last lines of each log:"
  for log in "$PARTS"/*.log; do echo "--- $log"; tail -3 "$log"; done
  exit 1
fi

parts=("$BASE")
for month in $MONTHS; do parts+=("$PARTS/$month"); done
rm -rf "$OUT"
$PY scripts/merge_corpora.py --out "$OUT" "${parts[@]}"
$PY scripts/build_splits.py --corpus "$OUT"
echo "v2 corpus ready in $OUT"
