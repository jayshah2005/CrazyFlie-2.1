#!/usr/bin/env bash
# Train the brushless policy and run the robustness check. Runs in WSL Ubuntu or
# on macOS (platform.sh) on the training code in ../learning_to_fly (submodules initialised, see ../README.md).
#
#   STEPS=1000001     training steps (checkpoint every 100k)
#   ANG_ACC=2e-4      angular-acceleration penalty (0 = paper reward)
#   REAL_NOISE=1      measured sensor noise (0 = paper noise)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/platform.sh"
L2F=${L2F:-$HERE/../learning_to_fly}
STEPS=${STEPS:-1000001}
ANG_ACC=${ANG_ACC:-2e-4}
REAL_NOISE=${REAL_NOISE:-1}
FLAGS="-DL2F_BRUSHLESS -DL2F_STEP_LIMIT=$STEPS -DL2F_ANG_ACC_WEIGHT=$ANG_ACC"
[ "$REAL_NOISE" = 1 ] && FLAGS="$FLAGS -DL2F_REAL_NOISE"

cd "$L2F"

BUILD=build_brushless_${STEPS}_acc${ANG_ACC}_noise${REAL_NOISE}
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release "${CMAKE_PLATFORM[@]}" \
  -DCMAKE_CXX_FLAGS="$FLAGS" > /dev/null
cmake --build "$BUILD" -j"$NPROC" --target training_headless evaluate_robustness

export OPENBLAS_NUM_THREADS=1
"./$BUILD/src/training_headless" | grep -E "Step: .*00000 |Training took"
RUN=$(ls -td checkpoints/multirotor_td3/*/ | head -1)
LAST=$(ls "$RUN"actor_*.h5 | sort | tail -1)
echo "run: $RUN"
"./$BUILD/src/evaluate_robustness" "$LAST" 200
"./$BUILD/src/evaluate_robustness" "$LAST" 1 probe | tail -6   # motor-command jumpiness in hover
echo "firmware: ACTOR=${LAST%.h5}.h bash $HERE/../firmware/build_firmware.sh"
