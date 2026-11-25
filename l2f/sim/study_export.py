"""Export an ablation study's results to CSV: the summary table and curves over training steps.

Reads a study output folder (l2f/learning_to_fly/ablation_<study>_<time>, written by ablation.py) and
writes into --out:
  summary.csv   one row per arm x test (medians over seeds and late checkpoints, as ablation.py
                prints them): circle, uneven, crashed, jitter, crash-free checkpoints; waypoint
                tests: crashed, arrival within 5 / 10 cm and its time, overshoot, final error
  per_seed.csv  the same mission-test numbers per seed
  curves.csv    long format (arm, seed, step, metric, value):
                  return            mean training return per 100k steps (train.py log)
                  <world>_<metric>  checkpoint tests 1.0-2.0M (mission test, plain world, 12 ms):
                                    world clean / randomized / jumps, metric crash / circle / jitter
                  cos_<a>_<b>, weight_<g>, norm_<g>, step_norm
                                    MOO gradient log (logs/train_moo_*.csv), mean per 50k steps
Usage: python l2f/sim/study_export.py --study-dir l2f/learning_to_fly/ablation_split_agg_20261124 \
         --out l2f/docs/results/split_agg
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_log  # noqa: E402

log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()

NUM = r'([-+]?[0-9.]+)'
MISSION = re.compile(r'test (\w+) +(\d+) ms \| arm (\S+) +\| (\S+) +circle +' + NUM + r' cm +uneven +' + NUM +
                     r' cm +crashed +' + NUM + r'% +jitter +' + NUM + r' +no-crash checkpoints (\d+)/(\d+)')
WAYPOINT = re.compile(r'waypoint (\S+) +\| arm (\S+) +\| crashed +' + NUM + r'%.*arrived 5 cm +' + NUM + r'% in +' + NUM +
                      r' s, 10 cm +' + NUM + r'% in +' + NUM + r' s +overshoot +' + NUM + r' cm +final error +' + NUM +
                      r' cm +peak speed +' + NUM)
RETURN = re.compile(r'step (\d+): mean training return ' + NUM)
CHECKPOINT = re.compile(r'\| INFO \| (\d{6,8}) +(.*)$')
WORLD = re.compile(r'crash +(\d+) +circle +' + NUM + r' cm +jitter +' + NUM)
BIN = 50_000


def summary_rows(log_text):
    """(summary rows, per-seed rows) from ablation.py's printed summary."""
    summary, per_seed, seen = [], [], set()
    for line in log_text.splitlines():
        m = MISSION.search(line)
        if m:
            cond, delay, arm, which, circle, uneven, crashed, jitter, ok, n = m.groups()
            row = [arm, f'{cond}_{delay}ms', circle, uneven, crashed, jitter, f'{ok}/{n}']
            if which == 'ALL':
                if (arm, row[1]) not in seen:
                    seen.add((arm, row[1]))
                    summary.append(row + [''] * 7)
            else:
                seed = re.search(r'_s(\d+)_', which).group(1)
                per_seed.append([arm, seed] + row[1:])
            continue
        m = WAYPOINT.search(line)
        if m:
            test, arm, crashed, a5, t5, a10, t10, over, final, speed = m.groups()
            summary.append([arm, test, '', '', crashed, '', '', a5, t5, a10, t10, over, final, speed])
    return summary, per_seed


def gradient_curves(path):
    """Mean of every numeric column of a MOO gradient log per BIN steps: {(step, metric): value}."""
    sums, counts = defaultdict(float), defaultdict(int)
    with open(path) as f:
        for r in csv.DictReader(line for line in f if not line.startswith('#')):
            b = int(float(r['step']) // BIN) * BIN + BIN // 2
            for key, val in r.items():
                if key.startswith(('cos_', 'weight_', 'norm_')) or key == 'step_norm':
                    sums[(b, key)] += float(val)
                    counts[(b, key)] += 1
    return {k: sums[k] / counts[k] for k in sums}


def curve_rows(study_dir):
    """Long-format curves for every run in the study folder."""
    rows = []
    for train_out in sorted(study_dir.glob('*_s[0-9]*.txt')):
        m = re.match(r'(.+)_s(\d+)\.txt$', train_out.name)
        if not m:
            continue
        arm, seed = m.groups()
        text = train_out.read_text(errors='replace')
        rows += [[arm, seed, step, 'return', val] for step, val in RETURN.findall(text)]
        g = re.search(r'MOO gradient log: (\S+)', text)
        if g and (ROOT / g.group(1)).exists():
            for (step, key), val in sorted(gradient_curves(ROOT / g.group(1)).items()):
                rows.append([arm, seed, step, key, f'{val:.5g}'])
        test = study_dir / 'eval' / f'{arm}_s{seed}_plain_d12.txt'
        if test.exists():
            for line in test.read_text(errors='replace').splitlines():
                c = CHECKPOINT.search(line)
                if not c:
                    continue
                for world, part in zip(['clean', 'randomized', 'jumps'], c.group(2).split('|')):
                    w = WORLD.search(part)
                    if w:
                        for metric, val in zip(['crash', 'circle', 'jitter'], w.groups()):
                            rows.append([arm, seed, c.group(1), f'{world}_{metric}', val])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--study-dir', type=Path, required=True, help='ablation.py output folder')
    ap.add_argument('--log', default='ablation_log.txt', help='summary log inside the folder')
    ap.add_argument('--out', type=Path, required=True, help='folder for the CSV files')
    a = ap.parse_args()
    summary, per_seed = summary_rows((a.study_dir / a.log).read_text(errors='replace'))
    curves = curve_rows(a.study_dir)
    meta = dict(script=__file__, generated_at=STARTED_AT)
    run_log.write_csv(a.out / 'summary.csv', ['arm', 'test', 'circle_cm', 'uneven_cm', 'crashed_pct', 'jitter',
                                              'crash_free_checkpoints', 'arrive5_pct', 'arrive5_s', 'arrive10_pct',
                                              'arrive10_s', 'overshoot_cm', 'final_error_cm', 'peak_speed'], summary, **meta)
    run_log.write_csv(a.out / 'per_seed.csv', ['arm', 'seed', 'test', 'circle_cm', 'uneven_cm', 'crashed_pct', 'jitter',
                                               'crash_free_checkpoints'], per_seed, **meta)
    run_log.write_csv(a.out / 'curves.csv', ['arm', 'seed', 'step', 'metric', 'value'], curves, **meta)
    log.info('%d summary rows, %d per-seed rows, %d curve points -> %s', len(summary), len(per_seed), len(curves), a.out)


if __name__ == '__main__':
    main()
