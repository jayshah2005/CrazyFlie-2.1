#!/usr/bin/env bash
# Re-test the runs of a compare.sh folder at several strengths of the random external force
# (plan Q16). train_mission.sh tests each run at the force it trained with (its FORCE_SCALE),
# so runs trained at different forces would get different tests. This runs the same mission
# test (the settings of train_mission.sh's ev(); keep the two in step) on checkpoints 1.2M-2.0M
# of every run at each level, then summarises each option set per level, pooled over seeds.
#
#   LEVELS="1 0.5 0.25 0"   force levels (default)
#   JOBS=3                  tests at the same time
#
#   bash l2f/sim/cross_eval.sh l2f/learning_to_fly/compare_force
#   -> <dir>/xeval/<name>_s<seed>_ev<level>.txt (train_mission.sh's row format)
#
# At a run's own level the rows equal train_mission.sh's (checked 2026-11-22). Crash shares
# count the clean and uneven-motor tests only: the sensor path has no 1% jumps.
set -euo pipefail
SIM=$(cd "$(dirname "$0")" && pwd)
L2F=$(cd "$SIM/../learning_to_fly" && pwd)
source "$SIM/platform.sh"  # first_number
EPISODES=15  # train_mission.sh's default

if [ "${1:-}" = one ]; then  # one run output file at one force level
  f=$2; level=$3
  line=$(grep -m1 'runtime options:' "$f")
  run=${line##*run: }
  opts=${line#*runtime options: }; opts=${opts%%, run:*}
  ENVW=()
  for w in $opts; do case $w in L2F_FORCE_SCALE=*|none) ;; *) ENVW+=("$w") ;; esac; done
  ENVW+=("L2F_FORCE_SCALE=$level")
  # the run folder name holds the build: <time>_<ablation>_<seed>_<build without build_>_s<seed>_...
  build=$(basename "$run" | sed -E 's#^.*_[0-9]{3}_(mission_.*)_s[0-9]+_.*$#build_\1#')
  [ -x "$L2F/$build/src/evaluate_robustness" ] || { echo "no $build for $f" >&2; exit 1; }
  ev(){
    out=$(cd "$L2F" && L2F_GLITCH_P=$3 L2F_MISSION_H=0.75 L2F_MISSION_R=0.6 L2F_MISSION_P=10 L2F_MISSION_V_DOWN=0.25 \
          L2F_EVAL_GE=1 L2F_EVAL_NOISE=1 L2F_DR_SCALE=0.5 L2F_EVAL_DR=$2 L2F_EVAL_DELAY_MS=6 L2F_EVAL_KI=0 \
          L2F_MOTOR_GAIN=0.05 L2F_MOTOR_OFFSET=500 \
          env "${ENVW[@]}" "./$build/src/evaluate_robustness" "$1" "$EPISODES" mission 2>/dev/null)
    printf "crash %2s circle %5s cm jitter %s" "$(first_number "$out" 'crashed ')" \
      "$(first_number "$out" 'circle +tracking error mean +')" "$(first_number "$out" 'metric\) ')"
  }
  dst="$(dirname "$f")/xeval/$(basename "$f" .txt)_ev$level.txt"
  {
    echo "build $build, run $run, test options: ${ENVW[*]}"
    echo "checkpoint   clean                             | randomized                        | jumps 1%"
    for ck in $(ls "$L2F/$run"actor_*.h5 | sort | tail -9); do
      echo "$(basename "$ck" .h5 | sed 's/actor_0*//')  $(ev "$ck" 0 0) | $(ev "$ck" 1 0) | $(ev "$ck" 0 0.01)"
    done
  } > "$dst.tmp"
  mv "$dst.tmp" "$dst"
  exit 0
fi

OUT=$(cd "${1:?usage: cross_eval.sh <compare.sh output folder>}" && pwd)
LEVELS=${LEVELS:-1 0.5 0.25 0}
JOBS=${JOBS:-3}
mkdir -p "$OUT/xeval"
for f in "$OUT"/*_s[0-9].txt; do
  for level in $LEVELS; do printf '%s %s\n' "$f" "$level"; done
done | xargs -P "$JOBS" -n 2 bash "$0" one

names=$(ls "$OUT"/*_s[0-9].txt | sed -E 's#.*/(.*)_s[0-9]+\.txt#\1#' | sort -u)
for level in $LEVELS; do
  echo "== tested at force $level"
  for name in $names; do
    python3 "$SIM/moo_summary.py" checkpoints --no-jumps "$OUT/xeval/${name}"_s*_ev"$level".txt 2>&1 \
      | grep ' ALL ' | sed -E "s/^.*\| INFO \| ALL +/$(printf '%-10s' "$name") /"
  done
done
