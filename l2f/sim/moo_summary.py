"""Summaries of multi-objective (MOO=1) training runs.

checkpoints  medians over the late checkpoints of train_mission.sh outputs (the table it
             prints after training): circle error clean / with uneven motors, crash share,
             jitter and the number of checkpoints without any crash; per file and pooled.
             --no-jumps leaves the 1%-jumps test out of the crash counts (on the sensor path,
             SENSOR_SIM=1, it has no jumps: the estimator makes its own errors).
               python l2f/sim/moo_summary.py checkpoints out_s0.txt out_s1.txt out_s2.txt

conflicts    gradient conflicts from a run's MOO logs ($BUILD/moo_seed$SEED.csv, and if
             present moo_noise_seed$SEED.csv / moo_gram_seed$SEED.csv), per training window:
             median gradient norm and weight per objective, share of objective pairs x updates
             with cosine < 0, the most conflicting pairs; across two independent minibatches
             (noise log): each objective's own agreement and the cross-batch pair cosines.
             For reward parts (framing A) also the conflicts between groups of parts
             (tracking, smoothness, safety), computed from the Gram matrices.
               python l2f/sim/moo_summary.py conflicts l2f/learning_to_fly/build_mission_E_2000001_moo_tot1
"""
from __future__ import annotations

import argparse
import csv
import math
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
WINDOWS = [(0, 400_000), (400_000, 800_000), (800_000, 1_200_000), (1_200_000, 1_600_000), (1_600_000, 2_000_001)]
PARTS = ['survival', 'position', 'orientation', 'linear_velocity', 'action', 'action_rate',
         'angular_acceleration', 'landing_speed']
GROUPS = {'tracking': [1, 3], 'smoothness': [4, 5, 6], 'safety': [0, 2, 7]}


def checkpoint_rows(path: Path, first: int) -> list[tuple[float, ...]]:
    """Parse the checkpoint table printed by train_mission.sh, keeping checkpoints from step `first` on."""
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


def cmd_checkpoints(args: argparse.Namespace) -> None:
    """`checkpoints` subcommand: per-file and pooled late-checkpoint medians."""
    pooled = []
    tests = 2 if args.no_jumps else 3
    for path in args.files:
        rows = checkpoint_rows(path, args.first)
        if not rows:
            log.warning('%s: no checkpoint rows from step %d', path.name, args.first)
            continue
        pooled += rows
        log.info('%-40s %s', path.name, describe(rows, args.episodes, tests))
    if len(args.files) > 1 and pooled:
        log.info('%-40s %s', 'ALL', describe(pooled, args.episodes, tests))


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of dicts."""
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def in_window(rows: list[dict[str, str]], lo: int, hi: int) -> list[dict[str, str]]:
    """Rows whose training step lies in [lo, hi)."""
    return [r for r in rows if lo <= int(r['step']) < hi]


def group_lines(gram_rows: list[dict[str, str]]) -> list[str]:
    """Median cosine between the gradient groups (tracking, smoothness, safety), from logged Gram matrices."""
    def cosine(r, A, B):
        g = lambda a, b: float(r[f'g_{min(a, b)}_{max(a, b)}'])
        ab = sum(g(i, j) for i in A for j in B)
        aa = sum(g(i, j) for i in A for j in A)
        bb = sum(g(i, j) for i in B for j in B)
        return ab / math.sqrt(aa * bb) if aa > 0 and bb > 0 else 0.0
    names = list(GROUPS)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = [cosine(r, GROUPS[a], GROUPS[b]) for r in gram_rows]
            out.append(f'{a}-{b} {st.median(c):+.2f} ({100 * sum(v < 0 for v in c) / len(c):.0f}% neg)')
    return out


def cmd_conflicts(args: argparse.Namespace) -> None:
    """`conflicts` subcommand: per-window gradient norms, weights and conflicts from a run's MOO logs."""
    main = read_csv(args.build / f'moo_seed{args.seed}.csv')
    noise_path = args.build / f'moo_noise_seed{args.seed}.csv'
    gram_path = args.build / f'moo_gram_seed{args.seed}.csv'
    noise = read_csv(noise_path) if noise_path.exists() else []
    objs = [k[5:] for k in main[0] if k.startswith('norm_')]
    parts = objs == PARTS
    gram = read_csv(gram_path) if parts and gram_path.exists() else []
    pairs = [(a, b) for i, a in enumerate(objs) for b in objs[i + 1:]]
    log.info('%s, seed %d: %d objectives (%s)', args.build.name, args.seed, len(objs),
             'reward parts' if parts else 'tasks')
    for lo, hi in WINDOWS:
        rows = in_window(main, lo, hi)
        if not rows:
            continue
        log.info('== %dk-%dk (%d logged updates)', lo // 1000, hi // 1000, len(rows))
        log.info('  norm  %s', ' '.join(f'{o}={st.median(float(r["norm_" + o]) for r in rows):.2f}' for o in objs))
        log.info('  w     %s', ' '.join(f'{o}={st.median(float(r["w_" + o]) for r in rows):.2f}' for o in objs))
        cos = {p: [float(r[f'cos_{p[0]}__{p[1]}']) for r in rows] for p in pairs}
        med = {p: st.median(v) for p, v in cos.items()}
        neg = {p: 100 * sum(x < 0 for x in v) / len(v) for p, v in cos.items()}
        log.info('  same batch: median cos %+.2f, lowest %+.2f, pairs x updates negative %.0f%%',
                 st.median(med.values()), min(med.values()), st.mean(neg.values()))
        for p in sorted(pairs, key=med.get)[:args.top]:
            log.info('    %s - %s  cos %+.2f  negative %.0f%%', p[0], p[1], med[p], neg[p])
        q = in_window(noise, lo, hi)
        if q:
            same = ' '.join(f'{o}={st.median(float(r["same_" + o]) for r in q):.2f}' for o in objs)
            cross = {p: st.median(float(r[f'cross_{p[0]}__{p[1]}']) for r in q) for p in pairs}
            log.info('  own agreement across batches  %s', same)
            log.info('  cross batch: median %+.2f, most conflicting %s', st.median(cross.values()),
                     ', '.join(f'{a}-{b} {cross[(a, b)]:+.2f}' for a, b in sorted(pairs, key=cross.get)[:args.top]))
        g = in_window(gram, lo, hi)
        if g:
            log.info('  groups: %s', ' | '.join(group_lines(g)))


def main() -> None:
    """Parse the subcommand and run it."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('checkpoints', help='late-checkpoint medians of train_mission.sh outputs')
    c.add_argument('files', type=Path, nargs='+')
    c.add_argument('--first', type=int, default=1_200_000, help='first checkpoint step counted')
    c.add_argument('--episodes', type=int, default=15, help='episodes per condition (EPISODES=)')
    c.add_argument('--no-jumps', action='store_true', help='count crashes of the clean and uneven-motor tests only')
    c.set_defaults(func=cmd_checkpoints)
    k = sub.add_parser('conflicts', help='gradient conflicts from a run\'s MOO logs')
    k.add_argument('build', type=Path, help='build folder holding moo_seed<SEED>.csv')
    k.add_argument('--seed', type=int, default=0)
    k.add_argument('--top', type=int, default=4, help='most conflicting pairs shown per window')
    k.set_defaults(func=cmd_conflicts)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
