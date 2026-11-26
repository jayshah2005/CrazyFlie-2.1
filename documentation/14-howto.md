<!-- group: Reference | reviewed: 2026-11-26 -->
# How-to

Commands for the common tasks, from the repository root. Each script's docstring has its full
options; run it with `--help`.

## Set up

```bash
# flight scripts
python3 -m venv .venv && source .venv/bin/activate
pip install cflib numpy                     # macOS also: brew install libusb
# training
pip install torch numpy pybind11 h5py scipy quadprog
git clone -b feat/pga-aggregators https://github.com/rkhosrowshahi/torchjd ../torchjd
pip install -e "../torchjd[quadprog_projector]"
git submodule update --init l2f/learning_to_fly/external/rl_tools l2f/learning_to_fly/controller
git -C l2f/learning_to_fly/external/rl_tools submodule update --init external/cli11 external/highfive external/json external/tensorboard
```

Toolchain: Ubuntu `cmake build-essential libhdf5-dev libopenblas-dev libboost-all-dev
protobuf-compiler libprotobuf-dev` (and `gcc-arm-none-eabi` for firmware); macOS Xcode tools,
Homebrew `cmake hdf5 boost protobuf` and Arm's GNU toolchain. `l2f/sim/platform.sh` picks Apple's
Accelerate on macOS. Details: [`l2f/README.md`](../l2f/README.md#setup).

## Build the simulator module

The old (truth) path with the current recipe's flags:

```bash
source l2f/sim/platform.sh
R="-DL2F_BRUSHLESS -DL2F_REAL_NOISE -DL2F_ANG_ACC_WEIGHT=2e-4 -DL2F_LANDING_SPEED_WEIGHT=5 -DL2F_POS_INTEGRAL
   -DL2F_MISSION -DL2F_STEP_LIMIT=2000001 -DL2F_REPLAY_CAP=700000 -DL2F_RPM_OBS -DL2F_HISTORY=0
   -DL2F_CRASH_PENALTY=200 -DL2F_SURVIVAL_BONUS=0 -DL2F_ACTION_RATE_WEIGHT=1"
cmake -S l2f/learning_to_fly -B l2f/learning_to_fly/build_py_E_nb200 -DCMAKE_BUILD_TYPE=Release \
  "${CMAKE_PLATFORM[@]}" -DL2F_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(which python3) -DCMAKE_CXX_FLAGS="$R"
cmake --build l2f/learning_to_fly/build_py_E_nb200 -j --target l2f_sim evaluate_robustness h5_to_header
```

Add `-DL2F_SENSOR_SIM` (and a separate build folder) for the sensor path. Checks:
`python3 l2f/py/test_sim.py`, `test_networks.py`, `test_td3.py`, `test_moo.py --build ...`.

## Train one run

The current recipe (the `common` lines of `l2f/sim/ablation_split_agg.txt`):

```bash
B=l2f/learning_to_fly/build_py_E_nb200/src
ENV="--env L2F_OMEGA_CRASH=34.9 --env L2F_GYRO_CLIP=34.9 --env L2F_DELAY_MAX_MS=20 --env L2F_DR_MOTOR_TAU=0.25
     --env L2F_MOTOR_GAIN_ALL=0.1 --env L2F_MOTOR_IDLE_RPM=2500
     --env L2F_DRAG_MAX=0.03 --env L2F_MOTOR_TAU_MIN=0.045 --env L2F_MOTOR_TAU_MAX=0.09 --env L2F_GAIN_WALK=0.03
     --env L2F_CRASH_PENALTY_OVERRIDE=400 --env L2F_POS_CAP=0.2 --env L2F_P_WAYPOINT=0.2 --env L2F_WAYPOINT_MAX=0.5
     --env L2F_WAYPOINT_S_MIN=0 --env L2F_WAYPOINT_S_MAX=0.5 --env L2F_W_TIME=0.5"
python3 l2f/py/train.py --build $B --variant E --seed 70 --evaluate $ENV                       # TD3 (variant E)
python3 l2f/py/train.py --build $B --variant E --seed 70 $ENV \
  --moo parts --parts 4 --total 1 --split-critic --agg svd_pga                                   # MOO
```

Checkpoints go to `l2f/learning_to_fly/checkpoints/multirotor_td3/<time>_py_E_..._<pid>/`, the MOO
gradient log to `logs/train_moo_*.csv`. One run uses one core. Studies 1-10 ran 8-12 at a time on a
MacBook Pro; on the 8 GB Mac keep to about 4, and use the lab computer for large studies.

## Test checkpoints

```bash
python3 l2f/py/train.py --build $B --evaluate-run <run folder> --episodes 100 --env L2F_EVAL_DELAY_MS=12
$B/evaluate_robustness <actor.h5> 100 mission          # also: waypoint, landing, probe
L2F_EVAL_LEG_S=3.5 L2F_WAYPOINT_MAX=0.5 $B/evaluate_robustness <actor.h5> 50 waypoint
```

## Run a study

A study file lists `common` options, `arm`s (each one change), `test` conditions and optional
`wtest` waypoint tests ([`l2f/sim/ablation.py`](../l2f/sim/ablation.py) header):

```bash
python3 l2f/sim/ablation.py --study l2f/sim/ablation_split_agg.txt --build $B \
  --seeds 70 71 72 73 74 75 76 77 --jobs 10
python3 l2f/sim/study_stats.py --study-dir l2f/learning_to_fly/ablation_split_agg_<time> \
  --baseline td3 --out l2f/docs/results/split_agg/stats.csv
python3 l2f/sim/study_export.py --study-dir l2f/learning_to_fly/ablation_split_agg_<time> --out l2f/docs/results/split_agg
```

Interrupted studies continue where they stopped. Outputs (`l2f/learning_to_fly/ablation_*`) and
checkpoints are git-ignored and stay on the machine that ran them.

## Check a reward before training

```bash
python3 l2f/py/reward_check.py --build $B --actor <actor.h5> $ENV [--candidate crash=400 ...] [--group-scale ...]
```

FAIL above 5% crash-better states in any episode type: do not train that reward
([Reward](06-reward.md#the-reward-check)).

## Look inside a MOO run

```bash
python3 l2f/py/moo_report.py logs/train_moo_*.csv                    # cosines, weights, norms per window
python3 l2f/sim/critic_snapshot.py --out snap.json --at 200000 1200000 -- \
  --seed 70 --moo parts --parts 4 --total 1 --agg sum --split-critic $ENV   # is the critic collapsed?
```

## Build firmware and flash

```bash
$B/h5_to_header <actor.h5> actor.h                   # or use a folder in l2f/policies/
ACTOR=$PWD/l2f/policies/<folder>/actor.h HISTORY=0 RPM_OBS=1 \
  OFFICIAL_ZIP=<firmware-cf21bl-2026.08.zip> bash l2f/firmware/build_firmware.sh
python3 flash_cold.py ~/l2f_2026/build/firmware-cf21bl-2026.08-l2f.zip   # cold boot only, then power-cycle
```

## Fly

```bash
python3 lighthouse_check.py && python3 motor_check.py          # before flying
python3 rl_trigger.py --mode takeoff_and_switch                # first flight of a new policy
python3 rl_trigger.py --mode mission --height 0.75 --radius 0.6 --period 10 --descent-rate 0.25
python3 rl_trigger.py --mode mission --height 0.75 --waypoints "0.7,0;0.2,0" --dwell 3.5 --descent-rate 0.25 --slow-telemetry
python3 l2f/sim/flight_report.py --label <method> logs/rl_trigger_flight_<time>.csv
```

Enter starts and stops, Ctrl+C cuts the motors, Q quits on the floor; face the drone along +X
([Flights](11-flight.md#flying-with-rl_triggerpy)).

## Survey the sensors

```bash
python3 noise_survey.py still        # also: heights, hover, slide, logs (see its header)
```

Sensor-level values go into `config/sensors.yaml` with their measured ranges
([Sensor path](08-sensor-path.md#calibration)).

## Use the lab computer

`ssh lab` (Tailscale; it may ask to re-authenticate in a browser). Linux, 14 cores, no sudo.
Long studies run in `tmux`; keep the Mac for flights. Lab outputs of the first waypoint study are
in `~/crazyflie/l2f/learning_to_fly/compare_*` and scripts in `~/l2f-runs/`.

## Maintain these docs

```bash
python3 documentation/tools/build_docs.py          # rebuild assets/bundle.js and check everything
python3 documentation/tools/build_docs.py --check  # check only (fails if the bundle is stale)
python3 documentation/tools/check_widgets.py       # after changing widget math (needs node)
python3 -m http.server 8000                        # then open http://localhost:8000/documentation/
```

See [Conventions](CONVENTIONS.md).
