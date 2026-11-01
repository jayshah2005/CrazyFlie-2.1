#!/usr/bin/env bash
# Train the brushless policy and run the robustness check. Run inside WSL Ubuntu
# with learning-to-fly cloned at $L2F (submodules initialised, see ../README.md).
#
#   STEPS=1000001  training steps (checkpoint every 100k)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
L2F=${L2F:-$HOME/learning_to_fly}
STEPS=${STEPS:-1000001}

cd "$L2F"
if ! grep -q L2F_BRUSHLESS src/config/parameters.h; then
  git apply "$HERE/learning_to_fly-brushless.patch"
fi

BUILD=build_brushless_$STEPS
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_BACKEND_ENABLE_OPENBLAS:BOOL=ON \
  -DCMAKE_CXX_FLAGS="-DL2F_BRUSHLESS -DL2F_STEP_LIMIT=$STEPS" > /dev/null
cmake --build "$BUILD" -j"$(nproc)" --target training_headless evaluate_robustness

export OPENBLAS_NUM_THREADS=1
"./$BUILD/src/training_headless" | grep -E "Step: .*00000 |Training took"
RUN=$(ls -td checkpoints/multirotor_td3/*/ | head -1)
LAST=$(ls "$RUN"actor_*.h5 | sort | tail -1)
echo "run: $RUN"
"./$BUILD/src/evaluate_robustness" "$LAST" 200
echo "firmware: ACTOR=${LAST%.h5}.h bash $HERE/../firmware/build_firmware.sh"
