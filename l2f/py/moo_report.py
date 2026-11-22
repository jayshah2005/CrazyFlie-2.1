"""Summarise the per-objective gradient logs of PyTorch MOO runs (logs/train_moo_*.csv, l2f/py/moo.py).

Per training window: the median pairwise cosine of the objective gradients and how often it
is negative (a conflict), the median aggregator weight per objective, and the median gradient
norm relative to the largest. cos(g_a, g_b) = g_a . g_b / (|g_a| |g_b|): below 0 the two
objectives pull the actor in opposed directions; only then can an aggregator change more
than the step size. Also the check column (|J^T w - step| / |step|, must be ~1e-7).

Usage: python l2f/py/moo_report.py logs/train_moo_*.csv [--pairs 6]
"""
import argparse
import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for run_log
import run_log  # noqa: E402

log = run_log.setup(__file__)
WINDOWS = [(0, 400_000), (400_000, 800_000), (800_000, 1_200_000), (1_200_000, 1_600_000), (1_600_000, 2_000_001)]


def read(path):
    """Rows of a run_log CSV (skipping the # metadata lines) as dicts."""
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith('#')))


def summarise(path, n_pairs):
    """Log the window summaries of one run."""
    rows = read(path)
    if not rows:
        log.warning('%s: empty', path.name)
        return
    cols = rows[0].keys()
    names = [c[len('weight_'):] for c in cols if c.startswith('weight_')]
    pairs = [c for c in cols if c.startswith('cos_')]
    worst_check = max(float(r['check_rel_error']) for r in rows)
    log.info('%s: aggregator %s, %d objectives, %d log rows, worst check %.1e', path.name, rows[0]['aggregator'],
             len(names), len(rows), worst_check)
    for lo, hi in WINDOWS:
        sel = [r for r in rows if lo <= int(r['step']) < hi]
        if not sel:
            continue
        cos = {p: [float(r[p]) for r in sel] for p in pairs}
        med = {p: st.median(v) for p, v in cos.items()}
        neg = {p: sum(x < 0 for x in v) / len(v) for p, v in cos.items()}
        worst = sorted(pairs, key=lambda p: med[p])[:n_pairs]
        weights = {n: st.median(float(r[f'weight_{n}']) for r in sel) for n in names}
        norms = {n: st.median(float(r[f'norm_{n}']) for r in sel) for n in names}
        top = max(norms.values()) or 1.0
        log.info('  %4.1f-%3.1fM: most opposed %s', lo / 1e6, hi / 1e6,
                 ', '.join(f"{p[4:]} {med[p]:+.2f} ({100 * neg[p]:.0f}% <0)" for p in worst))
        log.info('             weights %s | norms %s', ' '.join(f'{n} {w:.2f}' for n, w in weights.items()),
                 ' '.join(f'{n} {v / top:.2f}' for n, v in norms.items()))


def main():
    """Summarise every given log."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('logs', nargs='+', type=Path)
    parser.add_argument('--pairs', type=int, default=6, help='most opposed pairs to show per window')
    args = parser.parse_args()
    for path in args.logs:
        summarise(path, args.pairs)


if __name__ == '__main__':
    main()
