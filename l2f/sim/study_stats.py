"""Significance tests for an ablation study: every arm against a baseline arm, per seed.

The unit is the seed: each seed's median over its late checkpoints (as ablation.py prints it), so
checkpoints of one run, which are correlated, are not counted as independent samples. For each arm
and metric: median difference to the baseline with a bootstrap 95% interval (resampling seeds), a
two-sided Mann-Whitney U test, Cliff's delta (effect size, -1..1, sign as the difference), and the
p value corrected with Holm over all arms x metrics against that baseline.

Metrics: mission test (--test, default plain_12ms) circle_cm, uneven_cm, crashed_pct, jitter from the
study's summary log; waypoint tests (wp05L, wp08L) arrival within 10 cm (%) and its time, from the
eval files (median over the checkpoints in each file).

Usage: python l2f/sim/study_stats.py --study-dir l2f/learning_to_fly/ablation_split_agg_20261124 \
         --baseline td3 --baseline moo4sum_split --out l2f/docs/results/split_agg/stats.csv
"""
from __future__ import annotations

import argparse
import re
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_log  # noqa: E402
from ablation import wp_rows  # noqa: E402
from study_export import summary_rows  # noqa: E402

log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()

# metric -> True if lower is better
MISSION = {'circle_cm': True, 'uneven_cm': True, 'crashed_pct': True, 'jitter': True}
WAYPOINT = {'arrived10': False, 'time10': True}


def per_seed(study_dir: Path, log_name: str, test: str) -> dict[str, dict[str, dict[int, float]]]:
    """{arm: {metric: {seed: value}}} for the mission metrics and the waypoint tests."""
    out: dict[str, dict[str, dict[int, float]]] = {}
    _, rows = summary_rows((study_dir / log_name).read_text(errors='replace'))
    for arm, seed, t, circle, uneven, crashed, jitter, _ok in rows:
        if t != test:
            continue
        for metric, val in zip(MISSION, (circle, uneven, crashed, jitter)):
            out.setdefault(arm, {}).setdefault(metric, {})[int(seed)] = float(val)
    for path in sorted((study_dir / 'eval').glob('*_wp.txt')):
        m = re.match(r'(.+)_s(\d+)_(\w+)_wp\.txt$', path.name)
        if not m or m.group(1) not in out:
            continue
        arm, seed, cond = m.group(1), int(m.group(2)), m.group(3)
        rows_ = [r for r in wp_rows(path) if not np.isnan(r['arrived10'])]
        if rows_:
            for metric in WAYPOINT:
                out[arm].setdefault(f'{cond}_{metric}', {})[seed] = st.median(r[metric] for r in rows_)
    return out


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) - P(a < b) over all pairs."""
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def bootstrap_ci(a: np.ndarray, b: np.ndarray, rng, n=10_000) -> tuple[float, float]:
    """95% interval of median(a) - median(b), resampling seeds within each arm."""
    ia = rng.integers(0, len(a), (n, len(a)))
    ib = rng.integers(0, len(b), (n, len(b)))
    d = np.median(a[ia], axis=1) - np.median(b[ib], axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def holm(p: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p values."""
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--study-dir', type=Path, required=True, help='ablation.py output folder')
    ap.add_argument('--log', default='ablation_log.txt', help='summary log inside the folder')
    ap.add_argument('--test', default='plain_12ms', help='mission test condition (as in the summary)')
    ap.add_argument('--baseline', action='append', required=True, help='baseline arm (repeatable)')
    ap.add_argument('--out', type=Path, help='CSV of all comparisons')
    a = ap.parse_args()
    data = per_seed(a.study_dir, a.log, a.test)
    rng = np.random.default_rng(0)
    fields = ['baseline', 'arm', 'metric', 'better', 'seeds', 'arm_median', 'base_median', 'diff', 'ci_low', 'ci_high',
              'cliffs_delta', 'p', 'p_holm', 'verdict']
    rows = []
    for base in a.baseline:
        family = []
        for arm in data:
            if arm == base:
                continue
            for metric, values in data[arm].items():
                base_vals = data[base].get(metric, {})
                seeds = sorted(set(values) & set(base_vals))
                if len(seeds) < 3:
                    continue
                x = np.array([values[s] for s in seeds])
                y = np.array([base_vals[s] for s in seeds])
                lower_better = MISSION.get(metric, WAYPOINT.get(metric.split('_', 1)[-1], True))
                lo, hi = bootstrap_ci(x, y, rng)
                p = float(mannwhitneyu(x, y, alternative='two-sided').pvalue)
                family.append([base, arm, metric, 'lower' if lower_better else 'higher', len(seeds),
                               float(np.median(x)), float(np.median(y)), float(np.median(x) - np.median(y)),
                               lo, hi, cliffs_delta(x, y), p])
        for row, p_adj in zip(family, holm([r[-1] for r in family])):
            diff, lower_better = row[7], row[3] == 'lower'
            good = diff < 0 if lower_better else diff > 0
            verdict = ('better' if good else 'worse') if p_adj < 0.05 else (
                ('better?' if good else 'worse?') if row[-1] < 0.05 else 'no clear difference')
            rows.append(row + [p_adj, verdict])
    for r in rows:
        log.info('vs %-16s %-22s %-18s %8.3f vs %8.3f  diff %+8.3f [%+.3f, %+.3f]  delta %+.2f  p %.4f  holm %.4f  %s',
                 r[0], r[1], r[2], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13])
    if a.out:
        fmt = [[f'{v:.4g}' if isinstance(v, float) else v for v in r] for r in rows]
        run_log.write_csv(a.out, fields, fmt, script=__file__, generated_at=STARTED_AT)
        log.info('%d comparisons -> %s', len(rows), a.out)


if __name__ == '__main__':
    main()
