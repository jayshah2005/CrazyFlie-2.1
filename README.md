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

Lighthouse geometry for different spaces lives under `config/`:

| Path | Use |
| --- | --- |
| `config/IH222A/` | IH222A room (`config.yaml`, `samples.yaml`) |
| `config/DroneLab/` | DroneLab (add files when ready) |
| `config/current` | Symlink to the active space |

Point tooling at `config/current/config.yaml` (or swap the symlink) so scripts stay space-agnostic.

## Platform notes

- **Linux**: set Crazyradio udev rules — see [USB permissions](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/usb_permissions/).
- **Windows**: install Crazyradio drivers via [Zadig](https://www.bitcraze.io/documentation/repository/crazyradio-firmware/master/building/usbwindows/).

## Docs

- [cflib user guides](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/)
- [Lighthouse getting started](https://www.bitcraze.io/documentation/tutorials/getting-started-with-lighthouse/)
