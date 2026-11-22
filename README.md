# Crazyflie 2.1 Brushless: scripts and a learned flight controller

This repository has two parts:

1. **Python scripts that fly or test a Crazyflie** over the radio with Bitcraze's `cflib`:
   flip and figure-8 demos, pre-flight checks, and system-identification flights.
   They run on the stock firmware.
2. **A learned flight controller** (`l2f/`). A small neural network is trained in simulation
   with reinforcement learning ([Learning to Fly in Seconds](https://arxiv.org/abs/2311.13081)),
   compiled into the drone's firmware, and flies the drone directly by setting the four motor
   speeds, with no PID controller. `rl_trigger.py` starts and supervises those flights.
   Most of the recent work is here; [`l2f/README.md`](l2f/README.md) is the full write-up.

The drone is a **Crazyflie 2.1 Brushless** (`cf21bl`) with a Lighthouse deck (and a Flow deck),
positioned by two Lighthouse base stations.

## Repository map

### Flying the learned controller

| File | What it does |
| --- | --- |
| `rl_trigger.py` | Starts and supervises learned-controller flights: pre-flight checks, hold-Enter dead-man switch, modes `takeoff_and_switch`, `land`, `mission`, ...; logs every flight. Windows only. |
| `l2f/` | Training code, simulator, firmware build, and results. Start at [`l2f/README.md`](l2f/README.md). |
| `poster/` | Research poster (LaTeX) and its figures. |

### Checks before or after flying

| File | What it does |
| --- | --- |
| `lighthouse_check.py` | Is the Lighthouse deck seeing both base stations, with calibration and geometry stored? |
| `still_test.py` | Leaves the drone still and measures state-estimate noise. |
| `motor_check.py` | Spins the motors at idle and flags one that is far below the others. |
| `rpm_quality.py` | Reads existing logs (no flying) and checks the measured motor RPM: invalid reads, noise, lag. |
| `noise_survey.py` | Sensor survey for the simulator's noise model: still, on spacers, PID hovers, pushed along a track, and Lighthouse jumps in flight logs. Writes `l2f/sim/noise_measured.json`. |

### Measuring the drone (system identification)

| File | What it does |
| --- | --- |
| `sysid_hover.py` | Stock-PID flight with height steps; logs motor commands next to measured RPM. |
| `sysid_analyze.py` | Fits the motor model (lag, command-to-RPM map, thrust constant) to that log. The simulator uses these numbers. |

### Demos on the stock firmware

| File | What it does |
| --- | --- |
| `flip.py` | Backflip with Lighthouse positioning. Run `--no-flip` first. |
| `tricks.py` | Show: figure 8, then backflip, then side flip, then land. Handles the Flow deck's bad estimate mid-flip. |

### Tools and shared code

| File | What it does |
| --- | --- |
| `run_log.py` | Shared console and CSV logging. **Every script uses it** (see Conventions). |
| `flash_cold.py` | Flashes a firmware zip with a cold boot, for when cfclient cannot reach the bootloader. Never warm-boot flash this drone. |
| `config/` | Lighthouse geometry per room (`config/current` links to the active one), `landing.yaml`: the mission landing values shared by `rl_trigger.py` and training, and `sensors.yaml`: the simulator's sensor priors (until the survey measures them). |

Local-only folders (git-ignored): `logs/` (every flight's CSV), `cache/` (cflib's radio cache),
and clones of Bitcraze repositories (`crazyflie-firmware/`, `crazyflie-lib-python/`,
`aideck-gap8-examples/`).

## How the learned controller fits together

```
sysid_hover.py ──> sysid_analyze.py ──> motor model in the simulator
                                                │
                       l2f/sim/train_mission.sh: train in simulation (TD3), evaluate checkpoints
                                                │
                       l2f/firmware/build_firmware.sh: compile the chosen network into firmware
                                                │
                       flash_cold.py (or cfclient cold boot): flash the drone
                                                │
                       lighthouse_check.py, motor_check.py: pre-flight checks
                                                │
                       rl_trigger.py --mode mission: fly, log to logs/
```

Training and the firmware build run on Linux (the team uses WSL Ubuntu 24.04); flying runs on
Windows. The steps, the commands, and the results of every experiment are in
[`l2f/README.md`](l2f/README.md).

## Setup

- Python **3.10+**
- A [Crazyradio](https://www.bitcraze.io/products/crazyradio-2-0/) (or USB) link to the drone
- macOS: `brew install libusb` (needed for Python 3.12+)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install cflib numpy
python -c "import cflib; print(cflib)"
```

`cflib` is not vendored. If you need to change the library itself, install it from source:
`git clone https://github.com/bitcraze/crazyflie-lib-python.git && pip install -e crazyflie-lib-python`
([install notes](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/install/)).

For training, also fetch the two submodules (once, from the repo root):

```bash
git submodule update --init l2f/learning_to_fly/external/rl_tools l2f/learning_to_fly/controller
```

The nested submodules and the toolchain are listed in [`l2f/README.md`](l2f/README.md#steps).

## Running a script

Run from the repo root (scripts write to `logs/` and `cache/` relative to it):

```bash
python flip.py --no-flip
```

The radio address defaults to `radio://0/80/2M/E7E7E7E7EA` (drone "EA") in every script; set the `CFLIB_URI`
environment variable to fly a different drone. Close cfclient first: only one program can use
the radio.

### The learned controller (`rl_trigger.py`)

The policy runs onboard, so the drone must run the learned-controller firmware (flash it with a
cold boot). `flip.py` and `tricks.py` need the official firmware again afterwards.

- Hold **Enter** to fly. Releasing it lands the drone in mission mode, or cuts the motors in the
  other modes. **Q** quits (on the floor), **Ctrl+C** is an emergency stop. Power-cycle the drone
  between flights.
- The policy turns to **yaw 0 of the Lighthouse frame**: face the drone along +X before take-off.
- Start with `python rl_trigger.py --mode takeoff_and_switch`. `python rl_trigger.py --help` lists
  every mode and option.

## Lighthouse configuration

| Path | Use |
| --- | --- |
| `config/IH222A/` | IH222A room (`config.yaml`, `samples.yaml`) |
| `config/DroneLab/` | DroneLab |
| `config/current` | Symlink to the active space |

Point tooling at `config/current/config.yaml` (or swap the symlink) so scripts stay room-agnostic.

## Conventions

- **Logging:** every script uses `run_log.py`. `log = run_log.setup(__file__)` tags each console
  line with the script name and time; CSVs go to `logs/{script}_{kind}_{YYYYmmdd_HHMMSS}.csv` via
  `run_log.csv_path` and `run_log.write_csv`. Details:
  [`.agents/skills/run-log/SKILL.md`](.agents/skills/run-log/SKILL.md).
- **Git:** commit messages are `type[/scope]: summary` with types `feat`, `fix`, `refactor` and
  `chore`; see [`.agents/skills/git/SKILL.md`](.agents/skills/git/SKILL.md).
- **Every script starts with a docstring** saying what it does, how to run it and what to watch
  out for. Read it before flying.

## Platform notes

- **Linux**: set the Crazyradio udev rules; see [USB permissions](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/usb_permissions/).
- **Windows**: install the Crazyradio drivers via [Zadig](https://www.bitcraze.io/documentation/repository/crazyradio-firmware/master/building/usbwindows/).

## Further reading

- [cflib user guides](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/)
- [Lighthouse getting started](https://www.bitcraze.io/documentation/tutorials/getting-started-with-lighthouse/)
