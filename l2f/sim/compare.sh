#!/usr/bin/env bash
# Compare option sets: run train_mission.sh for every set and seed, JOBS at a time,
# keep each run's output, and summarise each set with summarize_checkpoints.py
# (medians over the late checkpoints, pooled over the seeds).
#
#   CONFIGS=<file>  one option set per line: "name OPTION=value ...", e.g.
#                     base    VARIANT=E
#                     zigzag  VARIANT=E SMOOTHING=zigzag
#                   blank lines and lines starting with # are skipped
#   SEEDS="0 1 2"   seeds per set
#   JOBS=2          trainings at the same time (WSL has ~8 GB: at most 2)
#   OUT=<dir>       outputs (default ../learning_to_fly/compare_<time>)
#
#   CONFIGS=step1.txt bash l2f/sim/compare.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CONFIGS=${CONFIGS:?set CONFIGS to a file of option sets}
SEEDS=${SEEDS:-0 1 2}
JOBS=${JOBS:-2}
OUT=${OUT:-$HERE/../learning_to_fly/compare_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
cp "$CONFIGS" "$OUT/configs.txt"

names=()
while read -r name opts; do
  case "$name" in ''|'#'*) continue ;; esac
  names+=("$name")
  for seed in $SEEDS; do
    # at most JOBS at a time (polling: macOS's bash 3.2 has no wait -n)
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
    echo "start $name seed $seed: $opts"
    # shellcheck disable=SC2086  # opts are OPTION=value words on purpose
    env $opts SEED="$seed" bash "$HERE/train_mission.sh" > "$OUT/${name}_s${seed}.txt" 2>&1 &
    sleep 2
  done
done < "$CONFIGS"
wait || true  # a failed run shows in its output file; summarise the rest

echo "outputs in $OUT"
for name in "${names[@]}"; do
  echo "== $name: $(grep -h -o 'runtime options: [^,]*' "$OUT/${name}"_s*.txt | head -1)"
  python3 "$HERE/summarize_checkpoints.py" "$OUT/${name}"_s*.txt 2>&1 | grep -v 'run started' | sed 's/^.*| INFO | //'
done
