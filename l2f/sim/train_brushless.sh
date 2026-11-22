#!/usr/bin/env bash
# Train the brushless policy and run the robustness check. Run on macOS or
# Linux/WSL Ubuntu on the training code in ../learning_to_fly (submodules initialised, see ../README.md).
#
#   STEPS=1000001     training steps (checkpoint every 100k)
#   ANG_ACC=2e-4      angular-acceleration penalty (0 = paper reward)
#   REAL_NOISE=1      measured sensor noise (0 = paper noise)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
L2F=${L2F:-$HERE/../learning_to_fly}
STEPS=${STEPS:-1000001}
ANG_ACC=${ANG_ACC:-2e-4}
REAL_NOISE=${REAL_NOISE:-1}
FLAGS="-DL2F_BRUSHLESS -DL2F_STEP_LIMIT=$STEPS -DL2F_ANG_ACC_WEIGHT=$ANG_ACC"
[ "$REAL_NOISE" = 1 ] && FLAGS="$FLAGS -DL2F_REAL_NOISE"

cd "$L2F"

BUILD=build_brushless_${STEPS}_acc${ANG_ACC}_noise${REAL_NOISE}
# BLAS: Apple Accelerate on macOS (set by CMakeLists.txt), OpenBLAS elsewhere
if [ "$(uname)" = Darwin ]; then BLAS_FLAG=; JOBS=$(sysctl -n hw.ncpu)
else BLAS_FLAG=-DRL_TOOLS_BACKEND_ENABLE_OPENBLAS:BOOL=ON; JOBS=$(nproc); fi
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release $BLAS_FLAG \
  -DCMAKE_CXX_FLAGS="$FLAGS" > /dev/null
cmake --build "$BUILD" -j"$JOBS" --target training_headless evaluate_robustness

export OPENBLAS_NUM_THREADS=1
"./$BUILD/src/training_headless" | grep -E "Step: .*00000 |Training took"
RUN=$(ls -td checkpoints/multirotor_td3/*/ | head -1)
LAST=$(ls "$RUN"actor_*.h5 | sort | tail -1)
echo "run: $RUN"
"./$BUILD/src/evaluate_robustness" "$LAST" 200
"./$BUILD/src/evaluate_robustness" "$LAST" 1 probe | tail -6   # motor-command jumpiness in hover
echo "firmware: ACTOR=${LAST%.h5}.h bash $HERE/../firmware/build_firmware.sh"
