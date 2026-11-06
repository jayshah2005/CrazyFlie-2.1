#!/usr/bin/env bash
# Train the mission policy (take-off from the floor, moving targets, landing; no
# PID) and pick the best checkpoint with the mission evaluation. Run inside WSL
# Ubuntu with learning-to-fly cloned at $L2F (see ../README.md).
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
# WSL has ~8 GB: at most two of these trainings in parallel, started at least
# 30 s apart (run folders are named by start time).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
L2F=${L2F:-$HOME/learning_to_fly}
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
if ! grep -q L2F_MISSION src/config/parameters.h include/learning_to_fly/simulator/multirotor.h; then
  git apply "$HERE/learning_to_fly-brushless.patch"
fi

if [ "$VARIANT" = I1 ]; then BUILD=build_mission_${STEPS}; else BUILD=build_mission_${VARIANT}_${STEPS}; fi
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_BACKEND_ENABLE_OPENBLAS:BOOL=ON \
  -DCMAKE_CXX_FLAGS="$FLAGS" > /dev/null
cmake --build "$BUILD" -j"$(nproc)" --target training_headless evaluate_robustness

export OPENBLAS_NUM_THREADS=1
mkdir -p checkpoints/multirotor_td3
before=$(ls -d checkpoints/multirotor_td3/*/ 2>/dev/null || true)
L2F_MISSION_RAMP=800000 L2F_GLITCH_P=0.003 L2F_DR=1 L2F_DR_SCALE=0.3 \
L2F_MOTOR_GAIN=$TRAIN_GAIN L2F_MOTOR_OFFSET=$TRAIN_OFFSET L2F_SEED=$SEED \
  "./$BUILD/src/training_headless" > >(grep --line-buffered -E "Step: .*00000 |Training took") &
TRAIN_PID=$!
# this run's checkpoint folder: the one that appears after the start
RUN=
for _ in $(seq 120); do
  RUN=$(comm -13 <(echo "$before" | sort) <(ls -d checkpoints/multirotor_td3/*/ | sort) | head -1)
  [ -n "$RUN" ] && break
  sleep 1
done
echo "variant $VARIANT, seed $SEED, run: $RUN"
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
  printf "crash %2s circle %5s cm jitter %s" "$(echo "$out" | grep -oP 'crashed \K[0-9]+')" \
    "$(echo "$out" | grep -oP 'circle +tracking error mean +\K[0-9.]+')" "$(echo "$out" | grep -oP 'metric\) \K[0-9.]+')"
}
echo "checkpoint   clean                             | randomized                        | jumps 1%"
for ck in $(ls "$RUN"actor_*.h5 | sort | tail -11); do
  echo "$(basename "$ck" .h5 | sed 's/actor_0*//')  $(ev "$ck" 0 0) | $(ev "$ck" 1 0) | $(ev "$ck" 0 0.01)"
done
echo "firmware: ACTOR=<chosen checkpoint>.h bash $HERE/../firmware/build_firmware.sh"
