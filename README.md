# Crazyflie custom scripts

Python scripts for Crazyflie 2.1 (flip demos, lighthouse configs, etc.). The Bitcraze `cflib` package is **not** vendored in this repo — install it separately.

## Requirements

- Python **3.10+**
- A [Crazyradio](https://www.bitcraze.io/products/crazyradio-2-0/) (or USB) link to the drone
- macOS: `brew install libusb` (needed for Python 3.12+)

Use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

## Install cflib (recommended)

From [PyPI](https://pypi.org/project/cflib/):

```bash
pip install cflib
```

That is enough to run scripts in this repo (for example `flip.py`).

### Optional: editable install from source

Only if you need to modify the library itself:

```bash
git clone https://github.com/bitcraze/crazyflie-lib-python.git
cd crazyflie-lib-python
pip install -e .
```

Official install notes: [Bitcraze cflib installation](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/install/).

## Writing / running custom scripts

1. Activate your venv and confirm import works: `python -c "import cflib; print(cflib)"`
2. Copy an example pattern (connect → commander / high-level commander → land) or start from `flip.py`.
3. Set the radio URI if needed (default in `flip.py` is `radio://0/80/2M/E7E7E7E7E7`, or override with the env helper Bitcraze uses).
4. Run from the repo root:

```bash
python flip.py
```

Console and CSV logs go through `run_log.py`. Each line is tagged with the
script name and timestamp; CSVs land in `logs/` as
`{script}_{kind}_{YYYYmmdd_HHMMSS}.csv` with `# script:` / `# generated_at:`
header comments.

Lighthouse geometry for different spaces lives under `config/`:

| Path | Use |
| --- | --- |
| `config/IH222A/` | IH222A room (`config.yaml`, `samples.yaml`) |
| `config/DroneLab/` | DroneLab (add files when ready) |
| `config/current` | Symlink to the active space |

Point tooling at `config/current/config.yaml` (or swap the symlink) so scripts stay space-agnostic.

## Learned RL controller (`rl_trigger.py`)

Flies an end-to-end RPM policy from [Learning to Fly in Seconds](https://arxiv.org/abs/2311.13081)
([code](https://github.com/arplaboratory/learning-to-fly)) on the Crazyflie 2.1 Brushless.
Training, the brushless model and the firmware build are in [`l2f/`](l2f/README.md). The
policy runs onboard, so the drone must run that firmware (flash via cfclient cold boot);
`flip.py` needs the official firmware again afterwards.

- Hold **Enter** to fly; releasing it cuts the motors while the policy flies. **Q** quits,
  **Ctrl+C** is an emergency stop. Windows only. Power-cycle the drone between flights.
- The policy turns to **yaw 0 of the Lighthouse frame**: face the drone along +X before take-off.
- Start with `python rl_trigger.py --mode takeoff_and_switch`.

Supporting tools: `still_test.py` (estimator noise with the drone still),
`sysid_hover.py` + `sysid_analyze.py` (stock-PID flight and motor-model fit).

## Platform notes

- **Linux**: set Crazyradio udev rules — see [USB permissions](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/usb_permissions/).
- **Windows**: install Crazyradio drivers via [Zadig](https://www.bitcraze.io/documentation/repository/crazyradio-firmware/master/building/usbwindows/).

## Docs

- [cflib user guides](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/)
- [Lighthouse getting started](https://www.bitcraze.io/documentation/tutorials/getting-started-with-lighthouse/)
