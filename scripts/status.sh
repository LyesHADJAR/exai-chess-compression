#!/usr/bin/env bash
# One look at everything that may be running: the experiments, the bigger
# dataset build, and the machine. Safe to run any time, and changes nothing.
#
#   bash scripts/status.sh
cd ~/exai-chess-compression 2>/dev/null || { echo "~/exai-chess-compression not found"; exit 1; }
rule() { printf -- '------------------------------------------------------------\n'; }

echo "status at $(date '+%H:%M')"
rule
echo "EXPERIMENTS (runs/arms)"
if pgrep -f "scripts/run_wave.sh" > /dev/null; then echo "  the wave is RUNNING"; else echo "  the wave is not running"; fi
found=0
for log in runs/arms/*.log; do
  [ -e "$log" ] || continue
  name=$(basename "$log" .log)
  dir=runs/arms/$name
  found=1
  if [ -f "$dir/evaluation.json" ]; then
    state="DONE"
  elif grep -qE "Traceback|Error" "$log" 2> /dev/null && [ ! -f "$dir/results.json" ]; then
    state="FAILED  (look: tail -20 $log)"
  elif [ -f "$dir/results.json" ]; then
    state="training finished, now measuring"
  else
    state=$(grep -E '^step' "$log" 2> /dev/null | tail -1 \
      | awk '{printf "training: step %s of 8,000, %s pos/s, about %s min left", $2, $9, $11}')
    [ -n "$state" ] || state="starting"
  fi
  printf '  %-16s %s\n' "$name" "$state"
done
[ "$found" = 1 ] || echo "  no experiments yet"
if grep -q "^Rule: keep" wave_core.log 2> /dev/null; then
  echo "  ALL FINISHED. Send me: tail -25 ~/exai-chess-compression/wave_core.log"
fi
rule
echo "BIGGER DATASET (v2 corpus)"
if [ -f data/v2/split_labels.npy ]; then
  echo "  READY in data/v2"
elif pgrep -f build_v2_corpus > /dev/null; then
  echo "  building..."
  for log in data/v2_parts/*.log; do
    [ -e "$log" ] || continue
    month=$(basename "$log" .log)
    if grep -qE "Traceback|Incomplete" "$log"; then
      s="FAILED  (look: tail -5 $log)"
    elif grep -q "^Wrote" "$log"; then
      s="done"
    elif grep -qE "games +[0-9,]+ positions" "$log"; then
      s="$(grep -E "games +[0-9,]+ positions" "$log" | tail -1 | awk '{print $1}') of 350,000 games"
    else
      s="downloading"
    fi
    printf '  %-10s %s\n' "$month" "$s"
  done
elif [ -f build_v2.log ]; then
  echo "  not running and not finished. Last lines:"
  tail -3 build_v2.log | sed 's/^/    /'
else
  echo "  not started"
fi
rule
echo "MACHINE"
df -h ~ 2> /dev/null | awk 'NR==2 {print "  free disk:   " $4}'
command -v free > /dev/null && free -g | awk '/Mem:/ {print "  free memory: " $7 " GB"}'
command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
  --format=csv,noheader 2> /dev/null | sed 's/^/  GPU (use, memory): /'
exit 0
