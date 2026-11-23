"""Ablation study for the PyTorch trainer: train every arm over the seeds, test every run
under every test condition, and summarise each arm x condition with summarize_checkpoints.py.

One change at a time (l2f/docs/time_to_target_study.md, section 7): each arm differs from the
`base` arm in the options it adds, and every arm is tested under the same conditions, so the
effect of training with an option is not mixed up with the effect of testing with it.

Study file (blank lines and lines starting with # are skipped):
  common  <words>   options for every arm and every test (several lines are joined)
  arm     <name> <words>    one training arm: its options on top of `common`
  test    <name> <words>    one test condition: its options on top of `common`
  wtest   <name> <words>    one waypoint test condition (evaluate_robustness ... waypoint: target
                            jumps sampled with these options, e.g. L2F_WAYPOINT_MAX), optional
A word KEY=VALUE is a runtime option (train.py --env); any other word goes to train.py as is
(e.g. --moo parts --parts 3 --agg sum).

Every run is tested at each --delays estimate delay (ms), 100 episodes per checkpoint and test
(train.py --evaluate-run: the last 11 checkpoints, clean / uneven motors / 1% jumps).
Waypoint tests run on every checkpoint from 1.2M on (--wp-episodes each) and report medians over
seeds x checkpoints: crashes, legs arrived within 5 cm and within 10 cm (both below 0.2 m/s) and
the time to each, overshoot, final error, peak speed. L2F_EVAL_LEG_S=<s> in a wtest line makes both
legs that long (test only), so slow policies can settle before a leg ends. Finished runs and tests are skipped, so an interrupted study continues where it stopped.

Usage:
  python l2f/sim/ablation.py --study l2f/sim/ablation_simopts.txt \
      --build l2f/learning_to_fly/build_py_E_nb200/src --seeds 70 71 72 73 --jobs 8
"""
from __future__ import annotations

import argparse
import os
import re
import statistics as st
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_log  # noqa: E402

log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()
TRAIN = ROOT / 'l2f' / 'py' / 'train.py'
SUMMARIZE = ROOT / 'l2f' / 'sim' / 'summarize_checkpoints.py'
CHECKPOINTS = ROOT / 'l2f' / 'learning_to_fly' / 'checkpoints' / 'multirotor_td3'
OPTION = re.compile(r'^[A-Z][A-Z0-9_]*=')


def split_words(words: list[str]) -> tuple[list[str], list[str]]:
    """(runtime options KEY=VALUE, other train.py arguments)."""
    env = [w for w in words if OPTION.match(w)]
    return env, [w for w in words if not OPTION.match(w)]


def read_study(path: Path):
    """common words, {arm: words}, {test: words}, {waypoint test: words}, in file order."""
    common, arms, tests, wtests = [], {}, {}, {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        words = line.split()
        if not words or words[0].startswith('#'):
            continue
        kind = words[0]
        if kind == 'common':
            common += words[1:]
        elif kind in ('arm', 'test', 'wtest') and len(words) >= 2:
            {'arm': arms, 'test': tests, 'wtest': wtests}[kind][words[1]] = words[2:]
        else:
            raise SystemExit(f'{path}:{n}: expected "common ...", "arm <name> ..." or "test <name> ..."')
    if not arms or not tests:
        raise SystemExit(f'{path}: needs at least one arm and one test')
    return common, arms, tests, wtests


def env_args(options: list[str]) -> list[str]:
    """train.py --env arguments for runtime options."""
    return [a for o in options for a in ('--env', o)]


def finished(path: Path, marker: str) -> bool:
    """True if `path` exists and contains `marker` (a run or test that completed)."""
    return path.exists() and marker in path.read_text(errors='replace')


def run_folder(train_out: Path) -> Path | None:
    """The checkpoint folder named in a training output ('run <name>: ...')."""
    m = re.search(r'\| INFO \| run (\S+): ', train_out.read_text(errors='replace'))
    return CHECKPOINTS / m.group(1) if m else None


def call(cmd: list[str], out: Path) -> int:
    """Run cmd with its output in `out`; return the exit code."""
    with out.open('w') as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def main() -> None:
    """Parse the options, train, test and summarise."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--study', type=Path, required=True, help='study file (see the header)')
    ap.add_argument('--build', type=Path, required=True, help='build folder holding l2f_sim and evaluate_robustness')
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--steps', type=int, default=2_000_001, help='training steps (the recipe is validated at 2M)')
    ap.add_argument('--jobs', type=int, default=6, help='trainings or tests at the same time')
    ap.add_argument('--delays', type=int, nargs='+', default=[6, 12], help='test estimate delays (ms)')
    ap.add_argument('--episodes', type=int, default=100, help='test episodes per checkpoint and test')
    ap.add_argument('--wp-episodes', type=int, default=50, help='waypoint test episodes per checkpoint')
    ap.add_argument('--out', type=Path, help='output folder (default l2f/learning_to_fly/ablation_<study>_<time>)')
    a = ap.parse_args()

    common, arms, tests, wtests = read_study(a.study)
    out = a.out or ROOT / 'l2f' / 'learning_to_fly' / f'ablation_{a.study.stem}_{STARTED_AT:%Y%m%d_%H%M%S}'
    (out / 'eval').mkdir(parents=True, exist_ok=True)
    (out / 'study.txt').write_text(a.study.read_text())
    python = sys.executable
    log.info('study %s: arms %s, tests %s, seeds %s, %d steps, %d jobs, output %s',
             a.study, ', '.join(arms), ', '.join(tests), a.seeds, a.steps, a.jobs, out)

    # 1. training
    def train(arm: str, seed: int) -> None:
        dest = out / f'{arm}_s{seed}.txt'
        if finished(dest, 'Training took'):
            log.info('train %s s%d: already done', arm, seed)
            return
        env, flags = split_words(common + arms[arm])
        cmd = [python, str(TRAIN), '--build', str(a.build), '--seed', str(seed), '--steps', str(a.steps),
               *flags, *env_args(env)]
        log.info('train %s s%d: start', arm, seed)
        t0 = time.time()
        code = call(cmd, dest)
        log.log(20 if code == 0 else 40, 'train %s s%d: exit %d after %.1f min', arm, seed, code,
                (time.time() - t0) / 60)

    with ThreadPoolExecutor(a.jobs) as pool:
        list(pool.map(lambda job: train(*job), [(arm, s) for arm in arms for s in a.seeds]))

    # 2. tests: every run under every condition and delay
    def test(arm: str, seed: int, cond: str, delay: int) -> None:
        dest = out / 'eval' / f'{arm}_s{seed}_{cond}_d{delay}.txt'
        if finished(dest, 'checkpoint'):
            return
        folder = run_folder(out / f'{arm}_s{seed}.txt')
        if folder is None or not folder.exists():
            log.error('test %s s%d: no checkpoint folder (training failed?)', arm, seed)
            return
        env, _ = split_words(common + tests[cond])
        cmd = [python, str(TRAIN), '--build', str(a.build), '--evaluate-run', str(folder),
               '--episodes', str(a.episodes), *env_args(env + [f'L2F_EVAL_DELAY_MS={delay}'])]
        code = call(cmd, dest)
        if code:
            log.error('test %s s%d %s %d ms: exit %d (see %s)', arm, seed, cond, delay, code, dest)

    jobs = [(arm, s, c, d) for arm in arms for s in a.seeds for c in tests for d in a.delays]
    log.info('testing: %d runs x %d conditions x %d delays', len(arms) * len(a.seeds), len(tests), len(a.delays))
    with ThreadPoolExecutor(a.jobs) as pool:
        list(pool.map(lambda job: test(*job), jobs))

    # 3. summary per arm x condition x delay (medians over seeds x late checkpoints)
    for cond in tests:
        for delay in a.delays:
            for arm in arms:
                files = [str(out / 'eval' / f'{arm}_s{s}_{cond}_d{delay}.txt') for s in a.seeds]
                res = subprocess.run([python, str(SUMMARIZE), '--episodes', str(a.episodes), *files],
                                     capture_output=True, text=True, cwd=ROOT)
                lines = [ln.split('| INFO |')[-1].strip() for ln in res.stdout.splitlines() + res.stderr.splitlines()
                         if '| INFO |' in ln and ('.txt' in ln or ' ALL ' in ln)]
                if not lines:
                    log.warning('test %-6s %2d ms | arm %-8s | no summary (see eval files)', cond, delay, arm)
                # per seed too: a pooled median hides a collapsed seed (ablation_simopts: base s71)
                for line in lines:
                    collapsed = ' 100.0%' in line and not line.startswith('ALL')
                    log.log(30 if collapsed else 20, 'test %-7s %2d ms | arm %-8s | %s%s', cond, delay, arm, line,
                            '  <- COLLAPSED SEED' if collapsed else '')

    # 4. waypoint tests (optional): every checkpoint from 1.2M on, medians over seeds x checkpoints
    if wtests:
        evaluator = a.build / 'evaluate_robustness'
        # train.py's training settings, observation noise on (as the mission test)
        base = {**os.environ, 'L2F_MISSION_RAMP': '800000', 'L2F_GLITCH_P': '0', 'L2F_DR': '1', 'L2F_DR_SCALE': '0.3',
                'L2F_MOTOR_GAIN': '0.05', 'L2F_MOTOR_OFFSET': '500', 'L2F_EVAL_NOISE': '1'}

        def wtest(arm: str, seed: int, cond: str) -> None:
            dest = out / 'eval' / f'{arm}_s{seed}_{cond}_wp.txt'
            if finished(dest, 'END'):
                return
            folder = run_folder(out / f'{arm}_s{seed}.txt')
            if folder is None or not folder.exists():
                log.error('waypoint test %s s%d: no checkpoint folder', arm, seed)
                return
            env, _ = split_words(common + wtests[cond])
            run_env = {**base, **dict(o.split('=', 1) for o in env)}
            text = []
            for ck in sorted(folder.glob('actor_*.h5')):
                if int(ck.stem.split('_')[1]) < 1_200_000:
                    continue
                res = subprocess.run([str(evaluator), str(ck), str(a.wp_episodes), 'waypoint'], env=run_env,
                                     capture_output=True, text=True, cwd=ROOT)
                text.append(f'== {ck.stem}\n{res.stdout}')
            dest.write_text(''.join(text) + 'END\n')

        log.info('waypoint testing: %d runs x %d conditions', len(arms) * len(a.seeds), len(wtests))
        with ThreadPoolExecutor(a.jobs) as pool:
            list(pool.map(lambda job: wtest(*job), [(arm, s, c) for arm in arms for s in a.seeds for c in wtests]))
        for cond in wtests:
            for arm in arms:
                pooled, per_seed = [], []
                for s in a.seeds:
                    rows = wp_rows(out / 'eval' / f'{arm}_s{s}_{cond}_wp.txt')
                    pooled += rows
                    if rows:
                        per_seed.append(f's{s} {st.median(r["crash"] for r in rows):.0f}%')
                if not pooled:
                    log.warning('waypoint %-8s | arm %-14s | no results', cond, arm)
                    continue
                med = {k: st.median(r[k] for r in pooled) for k in pooled[0]}
                log.info('waypoint %-8s | arm %-14s | crashed %4.1f%% (%s)  arrived 5 cm %4.1f%% in %.2f s, '
                         '10 cm %4.1f%% in %.2f s  overshoot %4.1f cm  final error %4.1f cm  peak speed %.2f m/s',
                         cond, arm, med['crash'], ', '.join(per_seed), med['arrived'], med['time'], med['arrived10'],
                         med['time10'], med['overshoot'], med['final'], med['speed'])
    log.info('outputs in %s', out)


def wp_rows(path: Path) -> list[dict[str, float]]:
    """One dict per checkpoint of a waypoint test file (evaluate_robustness ... waypoint output)."""
    if not path.exists():
        return []
    rows = []
    for block in path.read_text(errors='replace').split('== ')[1:]:
        m1 = re.search(r'waypoint episodes (\d+): crashed (\d+); legs \d+ \(mean [\d.]+ m\), arrived \d+ \(([\d.]+)%\)', block)
        m2 = re.search(r'time to target ([-\d.]+) s, overshoot ([-\d.]+) cm, final error ([-\d.]+) cm, '
                       r'peak speed ([-\d.]+) m/s', block)
        m3 = re.search(r'waypoint 10 cm: arrived \d+ \(([\d.]+)%\), time to target ([-\d.]+) s', block)
        if m1 and m2:
            rows.append({'crash': 100 * int(m1.group(2)) / int(m1.group(1)), 'arrived': float(m1.group(3)),
                         'time': float(m2.group(1)), 'overshoot': float(m2.group(2)), 'final': float(m2.group(3)),
                         'speed': float(m2.group(4)),
                         # evaluators before the 10 cm line: nan, so the summary shows it is missing
                         'arrived10': float(m3.group(1)) if m3 else float('nan'),
                         'time10': float(m3.group(2)) if m3 else float('nan')})
    return rows


if __name__ == '__main__':
    main()
