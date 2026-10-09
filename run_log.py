"""
Shared logging for Crazyflie scripts.

Every console line and CSV file records which script produced it and when
the run started / the file was written.
"""
from __future__ import annotations

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path('logs')


def resolve_script(script: str | None = None) -> str:
    """Return the script file name (e.g. ``flip.py``)."""
    if script:
        return Path(script).name
    main = sys.modules.get('__main__')
    path = getattr(main, '__file__', None) if main else None
    if path:
        return Path(path).name
    return 'unknown'


def script_stem(script: str | None = None) -> str:
    """Return the script stem for filenames (e.g. ``flip``)."""
    return Path(resolve_script(script)).stem


def setup(script: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure a logger named after the calling script.

    Format: ``YYYY-MM-DD HH:MM:SS | script.py | LEVEL | message``
    """
    name = resolve_script(script)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    logger.addHandler(handler)

    started = datetime.now().astimezone()
    logger.info('run started (script=%s, started_at=%s)',
                name, started.isoformat(timespec='seconds'))
    return logger


def csv_path(kind: str = 'flight', script: str | None = None,
             when: datetime | None = None) -> Path:
    """
    Build ``logs/{script}_{kind}_{YYYYmmdd_HHMMSS}.csv``.

    Creates ``logs/`` if needed. Uses the script stem (``flip``, not ``flip.py``).
    """
    name = script_stem(script)
    stamp = (when or datetime.now()).strftime('%Y%m%d_%H%M%S')
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f'{name}_{kind}_{stamp}.csv'


def write_csv(path: Path | str, fieldnames: list[str], rows,
              script: str | None = None,
              generated_at: datetime | None = None) -> Path:
    """
    Write a CSV with metadata comment lines naming the script and time.

    Leading lines::

        # script: flip.py
        # generated_at: 2026-10-09T12:11:00-04:00
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = resolve_script(script)
    when = generated_at or datetime.now().astimezone()

    with path.open('w', newline='') as f:
        f.write(f'# script: {name}\n')
        f.write(f'# generated_at: {when.isoformat(timespec="seconds")}\n')
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    return path
