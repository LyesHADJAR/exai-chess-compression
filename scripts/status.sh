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
wave_running=0
if pgrep -f "scripts/run_wave.sh" > /dev/null; then
  wave_running=1
  echo "  the wave is RUNNING"
else
  echo "  the wave is not running"
fi
found=0
unfinished=0
for log in runs/arms/*.log; do
  [ -e "$log" ] || continue
  name=$(basename "$log" .log)
  dir=runs/arms/$name
  found=1
  step=$(grep -E '^step' "$log" 2> /dev/null | tail -1 | awk '{print $2}')
  if [ -f "$dir/evaluation.json" ]; then
    state="DONE"
  else
    unfinished=$((unfinished + 1))
    if grep -qE "Traceback|Error" "$log" 2> /dev/null && [ ! -f "$dir/results.json" ]; then
      state="FAILED  (look: tail -20 $log)"
    elif [ "$wave_running" = 0 ]; then
      # A line saying "training" from a run that is no longer running is a lie.
      if [ -f "$dir/results.json" ]; then state="STOPPED while measuring"
      elif [ -n "$step" ]; then state="STOPPED at step $step (nothing is running it)"
      else state="STOPPED before it began"; fi
    elif [ -f "$dir/results.json" ]; then
      state="training finished, now measuring"
    elif [ -n "$step" ]; then
      state=$(grep -E '^step' "$log" | tail -1 \
        | awk '{printf "training: step %s of 8,000, %s pos/s, about %s min left", $2, $9, $11}')
    else
      state="starting"
    fi
  fi
  printf '  %-16s %s\n' "$name" "$state"
done
[ "$found" = 1 ] || echo "  no experiments yet"
latest=$(ls -t wave_*.log 2> /dev/null | head -n 1)
if [ "$wave_running" = 0 ] && [ "$found" = 1 ]; then
  if [ "$unfinished" = 0 ]; then
    echo "  ALL FINISHED. Send me: tail -25 ~/exai-chess-compression/${latest:-wave_core.log}"
  else
    echo "  NOTHING IS RUNNING, but $unfinished experiment(s) did not finish. Start the wave again."
  fi
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
echo "ROUND TWO PREPARATION (mate scan and puzzles for the v2 corpus)"
if pgrep -f prepare_v2 > /dev/null; then
  echo "  running. Last line:"
  tail -n 1 prepare_v2.log 2> /dev/null | cut -c1-90 | sed 's/^/    /'
elif grep -q "^Ready" prepare_v2.log 2> /dev/null; then
  echo "  READY"
elif [ -f prepare_v2.log ]; then
  echo "  not running and not finished. Last lines:"
  tail -n 3 prepare_v2.log | cut -c1-90 | sed 's/^/    /'
else
  echo "  not started"
fi
rule
echo "MACHINE"
echo "  this server has been up since: $(ps -o lstart= -p 1 2> /dev/null | sed 's/^ *//')"
df -h ~ 2> /dev/null | awk 'NR==2 {print "  free disk:   " $4}'
# Inside a container, free(1) shows the whole host. The container's own limit
# lives in the cgroup files.
limit=""
if [ -r /sys/fs/cgroup/memory.max ]; then
  limit=$(cat /sys/fs/cgroup/memory.max); used=$(cat /sys/fs/cgroup/memory.current)
elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
  limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes); used=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes)
fi
if [ -n "$limit" ] && [ "$limit" != max ]; then
  awk -v u="$used" -v l="$limit" 'BEGIN { if (l < 1e12) printf "  memory:      %.0f GB used of the %.0f GB this server may use\n", u/1e9, l/1e9 }'
fi
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2> /dev/null)
case "$gpu" in *N/A* | *Insufficient* | "") ;; *) echo "  GPU (use, memory): $gpu" ;; esac
exit 0
