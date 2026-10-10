---
name: run-log
description: >-
  Use the shared run_log module for all script console output and CSV/flight
  logs. Apply when writing or editing Crazyflie scripts, adding print/logging,
  saving telemetry CSVs, or creating new flight/demo scripts.
---

# Shared run_log — do not reinvent logging

All project scripts **must** use [`run_log.py`](../../../run_log.py) for console
output and CSV/telemetry files. Do **not** add ad-hoc `print()`, raw
`logging.basicConfig`, handmade CSV writers, or new logger helpers.

## Required pattern (every script)

```python
from datetime import datetime
import run_log

log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()
```

- Pass `__file__` so every line is tagged with the script name.
- Replace `print(...)` with `log.info` / `log.warning` / `log.error`.
- See `flip.py` and `script.py` as reference.

## CSV / flight logs

```python
path = run_log.csv_path('flight', script=__file__, when=STARTED_AT)
run_log.write_csv(path, fieldnames, rows, script=__file__,
                  generated_at=datetime.now().astimezone())
```

- Files go under `logs/` as `{script}_{kind}_{YYYYmmdd_HHMMSS}.csv`.
- `write_csv` always writes `# script:` and `# generated_at:` header comments.
- Never write CSVs with bare `open` + `csv.writer` unless extending `run_log.py`.

## Do not

- `print()` for operator-facing messages
- New modules that wrap logging “just for this script”
- Log filenames without script stem + timestamp
- CSVs without script / generated_at metadata
- Logging only to stderr without going through `run_log.setup`

## Extending logging

If `run_log.py` is missing a feature (levels, file handlers, extra metadata),
**extend `run_log.py`** and keep call sites on that API. Do not fork a second
logger.
