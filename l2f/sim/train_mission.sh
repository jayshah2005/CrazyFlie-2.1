#!/usr/bin/env bash
# Train the mission policy (take-off from the floor, moving targets, landing; no
# PID) and pick the best checkpoint with the mission evaluation. Run on macOS or
# Linux/WSL Ubuntu on the training code in ../learning_to_fly (see ../README.md).
#
# Recipe of the flown policy "I1" (2026-11-04, checkpoint 1.4M):
#   compile flags  brushless model, measured noise, angular-acceleration 2e-4,
#                  action-rate 1, landing-speed 5, position-error integral input,
#                  mission episodes; 2M steps, replay buffer capped at 700k (RAM)
#   runtime        L2F_MISSION_RAMP=800000 (curriculum), L2F_GLITCH_P=0.003
#                  (Lighthouse-like jumps), L2F_DR=1 L2F_DR_SCALE=0.3 (mild
#                  domain randomization); delay 0-12 ms, ground effect, circles
#                  up to 1.2 m are on by default with L2F_MISSION
#
# VARIANT picks the actor's inputs (measured-RPM experiment, 2026-11-06):
#   I1  the flown recipe, unchanged (149 inputs)
#   A   I1 inputs, trained with per-motor mismatch (149)
#   B   + measured rotor speeds, 32-step action history (153)
#   C   + measured rotor speeds, 4-step action history (41)
#   D   no rotor speeds, 4-step action history (37)
#   E   + measured rotor speeds, no action history (25)
# A-D train with per-motor mismatch (each motor: gain 1 +- 5%, offset +-500 rpm,
# from rpm_quality.py), so the measured speeds carry information the action
# history does not. The "randomized" evaluation column includes the same
# mismatch for every variant, I1 too.
#
# TD3 returns swing a lot late in training: do not trust the last checkpoint,
# compare all of them (the table below) and build the firmware from the best.
#
#   VARIANT=I1      see above
#   SEED=0          training seed offset (same seed + recipe = identical run)
#   STEPS=2000001   training steps (checkpoint every 100k)
#   EPISODES=15     mission episodes per checkpoint and condition
#   MOO=1           multi-objective TD3 (L2F_MOO, src/moo/td3_moo.h): the critic
#                   predicts each reward part separately; the actor combines the
#                   per-part gradients (plain sum = ordinary TD3). Writes the
#                   per-part gradient norms and cosines to $BUILD/moo_seed$SEED.csv
#                   and the Gram matrices to $BUILD/moo_gram_seed$SEED.csv
#   AGG=sum         (MOO=1) gradient aggregator from src/moo/aggregators.h: sum,
#                   upgrad or aligned_mtl (with TASKS=1 also task_share: each task
#                   weighted by its replay-buffer share, i.e. plain TD3 split per
#                   task); anything else adds _$AGG to the build
#   TASKS=1         (MOO=1) framing B: one actor objective per mission episode type
#                   (free flight, take-off, landing, circle, line), each from its own
#                   minibatch; ordinary scalar critic. Adds _moo1_tasks to the build
#   TASKS=9         as TASKS=1 plus four harder episode types (figure-eight, fast
#                   circle, point-to-point, gust hover); every mission type 10% of
#                   the episodes, free flight 20%. Adds _moo1_tasks9 to the build
#   TASKS=regime    as TASKS=1 but the 9 objectives are hidden dynamics regimes (delay
#                   0-7 / 7-13 / 13-20 ms x mass -20..-7 / -7..+7 / +7..+20 %), with
#                   those wider ranges in training (L2F_DELAY_MAX_MS=20, the cap of the
#                   delay model; L2F_DR_MASS=0.2). Adds _moo1_tasksR to the build
#   HARD=1          the harder episode mix of TASKS=9 without task objectives (e.g.
#                   framing A with harder tracking). Adds _hard to the build
#   REWARD=tuned    tuned: the hand-tuned weights and weight curriculum (all flown
#                   policies); uniform: every cost weight 1, no weight curriculum
#                   (untuned-reward experiment, adds _uniform to the build)
# Memory: one I1 training peaks at ~3 GB once the 700k replay buffer is full
# (measured on the 16 GB M6 Mac, 2026-11-22): at most 3 in parallel there, 2 under
# WSL (~8 GB). Start them at least 30 s apart (run folders are named by start time).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
L2F=${L2F:-$HERE/../learning_to_fly}
VARIANT=${VARIANT:-I1}
SEED=${SEED:-0}
STEPS=${STEPS:-2000001}
EPISODES=${EPISODES:-15}
FLAGS="-DL2F_BRUSHLESS -DL2F_REAL_NOISE -DL2F_ANG_ACC_WEIGHT=2e-4 -DL2F_ACTION_RATE_WEIGHT=1"
FLAGS="$FLAGS -DL2F_LANDING_SPEED_WEIGHT=5 -DL2F_POS_INTEGRAL -DL2F_MISSION"
FLAGS="$FLAGS -DL2F_STEP_LIMIT=$STEPS -DL2F_REPLAY_CAP=700000"
MOTOR_GAIN=0.05
MOTOR_OFFSET=500
case "$VARIANT" in
  I1) TRAIN_GAIN=0; TRAIN_OFFSET=0 ;;
  A)  TRAIN_GAIN=$MOTOR_GAIN; TRAIN_OFFSET=$MOTOR_OFFSET ;;
  B)  TRAIN_GAIN=$MOTOR_GAIN; TRAIN_OFFSET=$MOTOR_OFFSET; FLAGS="$FLAGS -DL2F_RPM_OBS" ;;
  C)  TRAIN_GAIN=$MOTOR_GAIN; TRAIN_OFFSET=$MOTOR_OFFSET; FLAGS="$FLAGS -DL2F_RPM_OBS -DL2F_HISTORY=4" ;;
  D)  TRAIN_GAIN=$MOTOR_GAIN; TRAIN_OFFSET=$MOTOR_OFFSET; FLAGS="$FLAGS -DL2F_HISTORY=4" ;;
  E)  TRAIN_GAIN=$MOTOR_GAIN; TRAIN_OFFSET=$MOTOR_OFFSET; FLAGS="$FLAGS -DL2F_RPM_OBS -DL2F_HISTORY=0" ;;
  *)  echo "unknown VARIANT=$VARIANT (I1, A, B, C, D, E)" >&2; exit 1 ;;
esac

cd "$L2F"

MOO=${MOO:-0}
if [ "$VARIANT" = I1 ]; then BUILD=build_mission_${STEPS}; else BUILD=build_mission_${VARIANT}_${STEPS}; fi
MOO_PARTS=${MOO_PARTS:-8}  # 1 = all parts merged (diagnostic: must match ordinary TD3)
# TASKS=1 (with MOO=1): framing B, the objectives are the 5 mission episode types; scalar critic
TASKS=${TASKS:-0}
TASK_ENV=()
if [ "$TASKS" = 1 ] || [ "$TASKS" = 9 ] || [ "$TASKS" = regime ]; then MOO_PARTS=1; fi
if [ "$TASKS" = regime ]; then TASK_ENV=(L2F_DELAY_MAX_MS=20 L2F_DR_MASS=0.2); fi
if [ "$TASKS" = 9 ] || [ "${HARD:-0}" = 1 ]; then
  TASK_ENV=(L2F_P_TAKEOFF=0.1 L2F_P_LAND=0.1 L2F_P_CIRCLE=0.1 L2F_P_LINE=0.1
            L2F_P_EIGHT=0.1 L2F_P_FAST_CIRCLE=0.1 L2F_P_P2P=0.1 L2F_P_GUST=0.1)
fi
if [ "$MOO" = 1 ]; then
  FLAGS="$FLAGS -DL2F_MOO -DL2F_MOO_PARTS=$MOO_PARTS"
  BUILD=${BUILD}_moo; [ "$MOO_PARTS" = 8 ] || BUILD=${BUILD}${MOO_PARTS}
  if [ "$TASKS" = 1 ]; then FLAGS="$FLAGS -DL2F_MOO_TASKS"; BUILD=${BUILD}_tasks; fi
  if [ "$TASKS" = 9 ]; then FLAGS="$FLAGS -DL2F_MOO_TASKS -DL2F_MOO_NTASKS=9"; BUILD=${BUILD}_tasks9; fi
  if [ "$TASKS" = regime ]; then FLAGS="$FLAGS -DL2F_MOO_TASKS -DL2F_MOO_NTASKS=9 -DL2F_TASK_REGIME"; BUILD=${BUILD}_tasksR; fi
  # MOO_NORM=1: outputs learn unweighted parts scaled to O(1); curriculum weights act outside
  if [ "${MOO_NORM:-0}" = 1 ]; then FLAGS="$FLAGS -DL2F_MOO_NORM"; BUILD=${BUILD}_norm; fi
  # MOO_TOTAL=<lambda>: critic loss = (total error)^2 + lambda * sum of (part error)^2
  if [ -n "${MOO_TOTAL:-}" ]; then FLAGS="$FLAGS -DL2F_MOO_TOTAL_LOSS=$MOO_TOTAL"; BUILD=${BUILD}_tot$MOO_TOTAL; fi
  AGG=${AGG:-sum}
  if [ "$AGG" != sum ]; then FLAGS="$FLAGS -DL2F_MOO_AGG=$AGG"; BUILD=${BUILD}_$AGG; fi
fi
if [ "${HARD:-0}" = 1 ]; then BUILD=${BUILD}_hard; fi
case "${REWARD:-tuned}" in
  tuned) ;;
  uniform) FLAGS="$FLAGS -DL2F_REWARD_UNIFORM"; BUILD=${BUILD}_uniform ;;
  *) echo "unknown REWARD=$REWARD (tuned, uniform)" >&2; exit 1 ;;
esac
# CRITIC_HIDDEN=<n>: critic hidden-layer width (default 64)
if [ -n "${CRITIC_HIDDEN:-}" ]; then FLAGS="$FLAGS -DL2F_CRITIC_HIDDEN=$CRITIC_HIDDEN"; BUILD=${BUILD}_ch$CRITIC_HIDDEN; fi
if [ "$MOO" = 1 ]; then
  export L2F_MOO_LOG="$L2F/$BUILD/moo_seed$SEED.csv" L2F_MOO_GRAM_LOG="$L2F/$BUILD/moo_gram_seed$SEED.csv"
  export L2F_MOO_NOISE_LOG="$L2F/$BUILD/moo_noise_seed$SEED.csv"
fi
# BLAS: Apple Accelerate on macOS (set by CMakeLists.txt), OpenBLAS elsewhere
if [ "$(uname)" = Darwin ]; then BLAS_FLAG=; JOBS=$(sysctl -n hw.ncpu)
else BLAS_FLAG=-DRL_TOOLS_BACKEND_ENABLE_OPENBLAS:BOOL=ON; JOBS=$(nproc); fi
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release $BLAS_FLAG \
  -DCMAKE_CXX_FLAGS="$FLAGS" > /dev/null
cmake --build "$BUILD" -j"$JOBS" --target training_headless evaluate_robustness

export OPENBLAS_NUM_THREADS=1
mkdir -p checkpoints/multirotor_td3
before=$(ls -d checkpoints/multirotor_td3/*/ 2>/dev/null || true)
L2F_MISSION_RAMP=800000 L2F_GLITCH_P=0.003 L2F_DR=1 L2F_DR_SCALE=0.3 \
L2F_MOTOR_GAIN=$TRAIN_GAIN L2F_MOTOR_OFFSET=$TRAIN_OFFSET L2F_SEED=$SEED \
  env ${TASK_ENV[@]+"${TASK_ENV[@]}"} "./$BUILD/src/training_headless" > >(grep --line-buffered -E "Step: .*00000 |Training took|L2F_MOO") &
TRAIN_PID=$!
# this run's checkpoint folder: the one that appears after the start
RUN=
for _ in $(seq 120); do
  RUN=$(comm -13 <(echo "$before" | sort) <(ls -d checkpoints/multirotor_td3/*/ | sort) | head -1)
  [ -n "$RUN" ] && break
  sleep 1
done
echo "variant $VARIANT, moo $MOO, seed $SEED, run: $RUN"
wait "$TRAIN_PID"
[ -n "$RUN" ] || { echo "no checkpoint folder appeared" >&2; exit 1; }

# Mission evaluation per checkpoint: 0.75 m, 0.6 m circle in 10 s, 6 ms delay,
# ground effect, no outer correction (ki 0); clean / half-paper randomization
# plus per-motor mismatch / 1% jumps per step. Lower circle error and jitter,
# 0 crashes = better.
ev(){
  out=$(L2F_GLITCH_P=$3 L2F_MISSION_H=0.75 L2F_MISSION_R=0.6 L2F_MISSION_P=10 L2F_MISSION_V_DOWN=0.25 \
        L2F_EVAL_GE=1 L2F_EVAL_NOISE=1 L2F_DR_SCALE=0.5 L2F_EVAL_DR=$2 L2F_EVAL_DELAY_MS=6 L2F_EVAL_KI=0 \
        L2F_MOTOR_GAIN=$MOTOR_GAIN L2F_MOTOR_OFFSET=$MOTOR_OFFSET \
        "./$BUILD/src/evaluate_robustness" "$1" "$EPISODES" mission 2>/dev/null)
  printf "crash %2s circle %5s cm jitter %s" "$(echo "$out" | sed -nE 's/.*crashed ([0-9]+).*/\1/p' | head -1)" \
    "$(echo "$out" | sed -nE 's/.*circle +tracking error mean +([0-9.]+).*/\1/p' | head -1)" "$(echo "$out" | sed -nE 's/.*metric\) ([0-9.]+).*/\1/p' | head -1)"
}
echo "checkpoint   clean                             | randomized                        | jumps 1%"
for ck in $(ls "$RUN"actor_*.h5 | sort | tail -11); do
  echo "$(basename "$ck" .h5 | sed 's/actor_0*//')  $(ev "$ck" 0 0) | $(ev "$ck" 1 0) | $(ev "$ck" 0 0.01)"
done
echo "firmware: ACTOR=<chosen checkpoint>.h bash $HERE/../firmware/build_firmware.sh"
