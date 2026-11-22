# Build settings per platform, sourced by train_brushless.sh and train_mission.sh.
#
#   Linux / WSL: OpenBLAS, as the team's runs have always used.
#   macOS:       Apple's Accelerate instead of OpenBLAS. Homebrew's Boost (1.89 and
#                later) no longer ships a boost_system library, which HighFive asks for,
#                so HighFive is built without Boost (rl_tools does not use HighFive's
#                Boost support). CMake 4 refuses the vendored projects' old minimum
#                version unless a policy floor is set.
#
# Accelerate and OpenBLAS round differently, so a macOS run is not bit-identical to
# the same recipe on WSL; compare them statistically (seeds, late-checkpoint medians).
if [ "$(uname -s)" = Darwin ]; then
  CMAKE_PLATFORM=(-DRL_TOOLS_BACKEND_ENABLE_ACCELERATE:BOOL=ON -DHIGHFIVE_USE_BOOST=OFF
                  -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
  NPROC=$(sysctl -n hw.ncpu)
else
  CMAKE_PLATFORM=(-DRL_TOOLS_BACKEND_ENABLE_OPENBLAS:BOOL=ON)
  NPROC=$(nproc)
fi

# first_number TEXT PREFIX: the number right after the first match of the extended
# regex PREFIX in TEXT (portable replacement for GNU grep -oP 'PREFIX\K[0-9.]+').
first_number() {
  printf '%s\n' "$1" | sed -nE "s/.*$2([0-9.]+).*/\1/p" | head -1
}
