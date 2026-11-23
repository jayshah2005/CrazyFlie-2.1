"""Late-checkpoint medians of training runs (the mission-test table that train_mission.sh and
l2f/py/train.py print after training): circle error clean and with uneven motors, crash share,
jitter, and the number of checkpoints without any crash; per file and pooled over the files.

--no-jumps leaves the 1%-jumps test out of the crash counts (the sensor path, SENSOR_SIM=1, has
no jumps: the estimator makes its own errors).

Usage: python l2f/sim/summarize_checkpoints.py [--no-jumps] out_s0.txt out_s1.txt out_s2.txt
"""
from __future__ import annotations

import argparse
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_log  # noqa: E402

log = run_log.setup(__file__)

# one checkpoint row of train_mission.sh or l2f/py/train.py (after run_log's prefix):
# clean | randomized (uneven motors) | 1% jumps
ROW = re.compile(r'^(?:.*\| INFO \| )?(\d{7})' + r'\s+\|?\s*crash\s+(\d+) circle\s+([\d.]+) cm jitter ([\d.]+)' * 3)


def checkpoint_rows(path: Path, first: int) -> list[tuple[float, ...]]:
    """Parse the checkpoint table in a run's output, keeping checkpoints from step `first` on."""
    rows = []
    for line in path.read_text(errors='replace').splitlines():
        m = ROW.match(line)
        if m and int(m.group(1)) >= first:
            rows.append(tuple(float(g) for g in m.groups()))
    return rows


def describe(rows: list[tuple[float, ...]], episodes: int, tests: int = 3) -> str:
    """One-line medians (circle error, uneven-motor error, crash share, jitter) over checkpoint rows.

    tests: how many of the tests (clean, uneven motors, 1% jumps) count for the crashes.
    """
    med = lambda i: st.median(r[i] for r in rows)
    crashes = [sum(r[1 + 3 * k] for k in range(tests)) for r in rows]
    share = 100 * sum(crashes) / (len(rows) * tests * episodes)
    return (f'circle {med(2):5.1f} cm  uneven {med(5):5.1f} cm  crashed {share:4.1f}%  '
            f'jitter {med(3):.3f}  no-crash checkpoints {sum(c == 0 for c in crashes)}/{len(rows)}')


def main() -> None:
    """Per-file and pooled late-checkpoint medians."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', type=Path, nargs='+')
    ap.add_argument('--first', type=int, default=1_200_000, help='first checkpoint step counted')
    ap.add_argument('--episodes', type=int, default=15, help='episodes per condition (EPISODES=)')
    ap.add_argument('--no-jumps', action='store_true', help='count crashes of the clean and uneven-motor tests only')
    args = ap.parse_args()
    tests = 2 if args.no_jumps else 3
    pooled = []
    for path in args.files:
        rows = checkpoint_rows(path, args.first)
        if not rows:
            log.warning('%s: no checkpoint rows from step %d', path.name, args.first)
            continue
        pooled += rows
        log.info('%-40s %s', path.name, describe(rows, args.episodes, tests))
    if len(args.files) > 1 and pooled:
        log.info('%-40s %s', 'ALL', describe(pooled, args.episodes, tests))


if __name__ == '__main__':
    main()
