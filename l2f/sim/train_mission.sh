#!/usr/bin/env bash
# Train the mission policy (take-off from the floor, moving targets, landing; no
# PID) and pick the best checkpoint with the mission evaluation. Runs in WSL
# Ubuntu or on macOS (platform.sh) on the training code in ../learning_to_fly
# (see ../README.md).
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
# A-E train with per-motor mismatch (each motor: gain 1 +- 5%, offset +-500 rpm,
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
#   MOO_TOTAL=1     (MOO=1) also train the critic on the total's TD error:
#                   loss = (total error)^2 + MOO_TOTAL * sum of (part error)^2.
#                   Needed for MOO runs to fly as well as the scalar critic
#   AGG=sum         (MOO=1) gradient aggregator from src/moo/aggregators.h and
#                   aggregators_pga.h: sum, upgrad, aligned_mtl, svd_pga or
#                   osnmf_pga (with TASKS=1 also task_share: each task
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
#   MOO_PARTS=8     (MOO=1) 1 = all reward parts merged into one critic output: the
#                   MOO code path learning the scalar reward (must match ordinary TD3)
#   MOO_NORM=1      (MOO=1) critic outputs learn unweighted parts scaled to O(1)
#                   (100% of episodes crashed in simulation)
#   CRITIC_HIDDEN=64  critic hidden-layer width
#   CRASH_PENALTY=0 reward -CRASH_PENALTY on a crash: more than 0.6 m from the
#                   target, touching the floor tilted > 60 deg, OMEGA_CRASH, or a
#                   failed landing (0 = paper reward, a crash scores 0). A landing is
#                   never penalized. Adds _crash<p> to the build
#   SURVIVAL_BONUS=2  the constant reward per flying step (paper: 2); 0 to test the
#                   crash penalty alone. Adds _bonus<c> to the build
#   SMOOTHING=rate  rate: action-rate penalty |a_t - a_t-1|^2, weight 1 (all flown
#                   policies); zigzag: |a_t - 2 a_t-1 + a_t-2|^2 with weight
#                   ZIGZAG_WEIGHT (default 0.25) instead (adds _zz<w>); none: neither
#                   (adds _nosmooth)
# Runtime options (no rebuild; the run folder name does not show them, the run's
# first line does; the checkpoint evaluation below uses them too):
#   GYRO_CLIP=<rad/s>    clip the actor's gyro reading at the sensor's range
#                        (34.9 = the firmware's +-2000 deg/s)
#   OMEGA_CRASH=<rad/s>  rotating faster than this on any axis is a crash (34.9)
#   LANDING_CUT=1        land like rl_trigger.py: motor cut near the floor ends the
#                        episode as a landing; values from config/landing.yaml
#   POS_NOISE=1          per-episode, per-axis position noise: ranges from
#                        noise_measured.json (the sensor survey) if present, else
#                        x, y 3-10 mm (z keeps the fixed 5 mm)
#   FORCE_SCALE=<x>      scale the per-episode random external force (sigma m g / 20
#                        per axis; 1 = the recipe). The flown estimator reads a sideways
#                        force as tilt, so with SENSOR_SIM=1 it sets part of the estimate error
#   SENSOR_SIM=1    (build) the sensor path (plan Step 2): the actor sees the flown Kalman
#                   estimator's output, fed by simulated IMU, range, flow and Lighthouse
#                   sensors (config/sensors.yaml, SENSORS=<file> to use another; the
#                   estimator is built from the firmware tree FW=, default
#                   ~/crazyflie-firmware-2026.08). The Gaussian observation noise is off
#                   (the sensors make the noise); not with POS_NOISE=1. Adds _ss
#   REWARD=tuned    tuned: the hand-tuned weights and weight curriculum (all flown
#                   policies); uniform: every cost weight 1, no weight curriculum
#                   (untuned-reward experiment, adds _uniform to the build)
# WSL has ~8 GB: at most two of these trainings in parallel. Each run's folder
# ends in a unique tag (build, seed, runtime options, process id), so parallel
# runs never share a folder.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/platform.sh"
L2F=${L2F:-$HERE/../learning_to_fly}
VARIANT=${VARIANT:-I1}
SEED=${SEED:-0}
STEPS=${STEPS:-2000001}
EPISODES=${EPISODES:-15}
FLAGS="-DL2F_BRUSHLESS -DL2F_REAL_NOISE -DL2F_ANG_ACC_WEIGHT=2e-4"
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
# CRASH_PENALTY=<p>: reward -p on a crash (termination); landings never terminate
if [ -n "${CRASH_PENALTY:-}" ]; then FLAGS="$FLAGS -DL2F_CRASH_PENALTY=$CRASH_PENALTY"; BUILD=${BUILD}_crash$CRASH_PENALTY; fi
# SURVIVAL_BONUS=<c>: constant reward per flying step (default 2)
if [ -n "${SURVIVAL_BONUS:-}" ]; then FLAGS="$FLAGS -DL2F_SURVIVAL_BONUS=$SURVIVAL_BONUS"; BUILD=${BUILD}_bonus$SURVIVAL_BONUS; fi
# SMOOTHING=rate|zigzag|none (default rate, the flown recipe)
case "${SMOOTHING:-rate}" in
  rate)   FLAGS="$FLAGS -DL2F_ACTION_RATE_WEIGHT=1" ;;
  zigzag) ZIGZAG_WEIGHT=${ZIGZAG_WEIGHT:-0.25}
          FLAGS="$FLAGS -DL2F_ACTION_RATE_WEIGHT=0 -DL2F_ZIGZAG_WEIGHT=$ZIGZAG_WEIGHT"; BUILD=${BUILD}_zz$ZIGZAG_WEIGHT ;;
  none)   FLAGS="$FLAGS -DL2F_ACTION_RATE_WEIGHT=0"; BUILD=${BUILD}_nosmooth ;;
  *) echo "unknown SMOOTHING=$SMOOTHING (rate, zigzag, none)" >&2; exit 1 ;;
esac
if [ "${SENSOR_SIM:-0}" = 1 ]; then FLAGS="$FLAGS -DL2F_SENSOR_SIM"; BUILD=${BUILD}_ss; fi
# CRITIC_HIDDEN=<n>: critic hidden-layer width (default 64)
if [ -n "${CRITIC_HIDDEN:-}" ]; then FLAGS="$FLAGS -DL2F_CRITIC_HIDDEN=$CRITIC_HIDDEN"; BUILD=${BUILD}_ch$CRITIC_HIDDEN; fi
if [ "$MOO" = 1 ]; then
  export L2F_MOO_LOG="$L2F/$BUILD/moo_seed$SEED.csv" L2F_MOO_GRAM_LOG="$L2F/$BUILD/moo_gram_seed$SEED.csv"
  export L2F_MOO_NOISE_LOG="$L2F/$BUILD/moo_noise_seed$SEED.csv"
fi
# runtime options, passed to training and to the checkpoint evaluation
RUN_ENV=()
if [ -n "${GYRO_CLIP:-}" ]; then RUN_ENV+=(L2F_GYRO_CLIP=$GYRO_CLIP); fi
if [ -n "${OMEGA_CRASH:-}" ]; then RUN_ENV+=(L2F_OMEGA_CRASH=$OMEGA_CRASH); fi
if [ "${LANDING_CUT:-0}" = 1 ]; then
  LANDING_FILE="$HERE/../../config/landing.yaml"
  landing(){ sed -nE "s/^$1:[[:space:]]*([0-9.]+).*/\1/p" "$LANDING_FILE"; }
  CUT_H=$(landing cut_height); CUT_SPREAD=$(landing cut_height_spread)
  RUN_ENV+=(L2F_LANDING_CUT=1
            L2F_CUT_H_MIN=$(awk "BEGIN { print $CUT_H - $CUT_SPREAD }") L2F_CUT_H_MAX=$(awk "BEGIN { print $CUT_H + $CUT_SPREAD }")
            L2F_CUT_WINDOW=$(landing cut_window) L2F_CUT_BELOW=$(landing below) L2F_FLOOR_TIMEOUT=$(landing floor_timeout)
            L2F_DESCENT_MIN=$(landing descent_rate_min) L2F_DESCENT_MAX=$(landing descent_rate_max))
fi
if [ -n "${FORCE_SCALE:-}" ]; then RUN_ENV+=(L2F_FORCE_SCALE=$FORCE_SCALE); fi
if [ "${SENSOR_SIM:-0}" = 1 ]; then
  if [ "${POS_NOISE:-0}" = 1 ]; then echo "SENSOR_SIM=1: the sensors make the noise, POS_NOISE=1 does not apply" >&2; exit 1; fi
  RUN_ENV+=(L2F_SENSORS=${SENSORS:-$HERE/../../config/sensors.yaml})
fi
if [ "${POS_NOISE:-0}" = 1 ]; then
  NOISE_FILE="$HERE/noise_measured.json"
  if [ -f "$NOISE_FILE" ]; then
    # position std range per axis from the sensor survey (noise_survey.py)
    for axis in x y z; do
      read -r lo hi < <(python3 -c "import json, sys; p = json.load(open(sys.argv[1]))['position']['$axis']; print(p['sigma_min'], p['sigma_max'])" "$NOISE_FILE")
      AX=$(echo "$axis" | tr xyz XYZ)
      RUN_ENV+=(L2F_POS_NOISE_MIN_$AX=$lo L2F_POS_NOISE_MAX_$AX=$hi)
    done
  else
    RUN_ENV+=(L2F_POS_NOISE_MIN_X=0.003 L2F_POS_NOISE_MAX_X=0.010 L2F_POS_NOISE_MIN_Y=0.003 L2F_POS_NOISE_MAX_Y=0.010)
  fi
fi
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release "${CMAKE_PLATFORM[@]}" ${FW:+-DFW="$FW"} \
  -DCMAKE_CXX_FLAGS="$FLAGS" > /dev/null
cmake --build "$BUILD" -j"$NPROC" --target training_headless evaluate_robustness

export OPENBLAS_NUM_THREADS=1
mkdir -p checkpoints/multirotor_td3
# unique run folder: build, seed, runtime options and this script's process id
TAG="${BUILD#build_}_s${SEED}"
for opt in GYRO_CLIP:gc OMEGA_CRASH:oc LANDING_CUT:cut POS_NOISE:pn; do
  var=${opt%%:*}; short=${opt#*:}
  if [ -n "${!var:-}" ] && [ "${!var}" != 0 ]; then TAG="${TAG}_${short}"; fi
done
if [ -n "${FORCE_SCALE:-}" ]; then TAG="${TAG}_fs${FORCE_SCALE}"; fi  # 0 is a setting too
TAG="${TAG}_$$"
L2F_MISSION_RAMP=800000 L2F_GLITCH_P=0.003 L2F_DR=1 L2F_DR_SCALE=0.3 \
L2F_MOTOR_GAIN=$TRAIN_GAIN L2F_MOTOR_OFFSET=$TRAIN_OFFSET L2F_SEED=$SEED L2F_RUN_TAG=$TAG \
  env ${TASK_ENV[@]+"${TASK_ENV[@]}"} ${RUN_ENV[@]+"${RUN_ENV[@]}"} "./$BUILD/src/training_headless" > >(grep --line-buffered -E "Step: .*00000 |Training took|L2F_MOO") &
TRAIN_PID=$!
# this run's checkpoint folder: the one ending in its tag
RUN=
for _ in $(seq 120); do
  # || true: before the folder exists, ls fails, which set -e / pipefail would end the script on
  RUN=$(ls -d checkpoints/multirotor_td3/*_"$TAG"/ 2>/dev/null | head -1 || true)
  [ -n "$RUN" ] && break
  sleep 1
done
echo "variant $VARIANT, moo $MOO, seed $SEED, smoothing ${SMOOTHING:-rate}, runtime options: ${RUN_ENV[*]:-none}, run: $RUN"
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
        env ${RUN_ENV[@]+"${RUN_ENV[@]}"} "./$BUILD/src/evaluate_robustness" "$1" "$EPISODES" mission 2>/dev/null)
  printf "crash %2s circle %5s cm jitter %s" "$(first_number "$out" 'crashed ')" \
    "$(first_number "$out" 'circle +tracking error mean +')" "$(first_number "$out" 'metric\) ')"
}
echo "checkpoint   clean                             | randomized                        | jumps 1%"
for ck in $(ls "$RUN"actor_*.h5 | sort | tail -11); do
  echo "$(basename "$ck" .h5 | sed 's/actor_0*//')  $(ev "$ck" 0 0) | $(ev "$ck" 1 0) | $(ev "$ck" 0 0.01)"
done
echo "firmware: ACTOR=<chosen checkpoint>.h bash $HERE/../firmware/build_firmware.sh"
