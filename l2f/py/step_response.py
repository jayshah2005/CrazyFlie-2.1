"""Target-step response of trained policies in the simulator (plan Q15: zig-zag vs action rate).

Point-to-point episodes only (the target jumps 0.2-0.45 m, mostly horizontally, 0.3-2 s into
the episode), deterministic actions, no domain randomization, no Lighthouse-like jumps. Per
episode, with d the step size and e(t) the distance to the new target:

  half      time from the step until e < d / 2
  overshoot max over the response of the position past the target along the step, / d
  settle    time from the step until e stays below max(0.1 d, 5 cm) (the policies keep a few cm
            of steady error, so a pure 10% band is often never reached)
  speed     the peak speed during the response
  chatter   mean |a_t - a_t-1| and mean |a_t - 2 a_t-1 + a_t-2|^2 (zig-zag cost) over the response

The response window is the 2 s after the step. Results are medians over episodes, per
checkpoint set. The action-rate penalty sum (a_t - a_t-1)^2 also taxes fast purposeful changes;
zig-zag taxes only reversals, so if it matters anywhere it should be here.

Usage: python l2f/py/step_response.py --build l2f/learning_to_fly/build_py_E/src \\
         --set rate=l2f/learning_to_fly/compare_step1/base_s*.txt --set zigzag=l2f/learning_to_fly/compare_step1/zigzag_s*.txt
       (each --set: name=output files of train_mission.sh runs; their late checkpoints are used)
"""
import argparse
import glob
import os
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # repo root, for run_log
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_log  # noqa: E402
from export import load_actor_h5  # noqa: E402
from networks import Actor  # noqa: E402

log = run_log.setup(__file__)
L2F = ROOT / 'l2f' / 'learning_to_fly'
DT = 0.01
WINDOW = 200  # steps after the step (2 s)


def checkpoints(outputs, steps):
    """The actor .h5 files of the given checkpoint steps for each train_mission.sh output file."""
    out = []
    for path in outputs:
        m = re.search(r'run: (\S+)', Path(path).read_text(errors='replace'))
        if not m:
            log.warning('%s: no run folder', path)
            continue
        for s in steps:
            ck = L2F / m.group(1) / f'actor_{s:015d}.h5'
            if ck.exists():
                out.append(ck)
    return out


def episode(sim, actor, info):
    """One point-to-point episode; the response metrics, or None if no step happened / it crashed."""
    sim.reset(np.array([True]))
    pos, acts, crashed = [], [], False
    for _ in range(info['step_limit']):
        obs, _ = sim.observe()
        with torch.no_grad():
            a = actor(torch.from_numpy(obs)).numpy()
        out = sim.step(a)
        pos.append(sim.states()['position'][0].copy())  # relative to the (stepping) target
        acts.append(a[0].copy())
        if out['terminated'][0]:
            crashed = not bool(out['landed'][0])
            break
    pos, acts = np.array(pos), np.array(acts)
    if len(pos) < 3:
        return {'crashed': crashed, 'valid': False, 'why': 'short'}
    jumps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    k = int(np.argmax(jumps)) + 1
    if jumps[k - 1] < 0.1:
        return {'crashed': crashed, 'valid': False, 'why': 'no step'}
    step = pos[k] - pos[k - 1]  # the relative position jumps by -(target offset)
    d = float(np.linalg.norm(step))
    u = -step / d               # direction of the target's move
    e = np.linalg.norm(pos[k:k + WINDOW], axis=1)
    past = pos[k:k + WINDOW] @ u  # p_rel = p - target starts at -d along u; > 0 = beyond the new target
    rise = next((i for i, x in enumerate(e) if x < 0.5 * d), None)
    above = np.nonzero(e >= max(0.1 * d, 0.05))[0]
    settle = (above[-1] + 1) if len(above) else 0
    vel = np.diff(pos[k - 1:k + WINDOW], axis=0) / DT
    vel[0] = 0  # the jump itself is not motion
    a = acts[k:k + WINDOW]
    chatter = float(np.mean(np.abs(np.diff(a, axis=0)))) if len(a) > 1 else float('nan')
    zz = float(np.mean(np.sum((a[2:] - 2 * a[1:-1] + a[:-2]) ** 2, axis=1))) if len(a) > 2 else float('nan')
    why = 'short response' if len(e) < WINDOW // 2 else ('no rise' if rise is None else '')
    return {'crashed': crashed, 'valid': not why, 'why': why, 'd': d,
            'rise': None if rise is None else rise * DT, 'overshoot': max(0.0, float(np.max(past))) / d,
            'settle': settle * DT if settle < len(e) else float('nan'), 'speed': float(np.max(np.linalg.norm(vel, axis=1))),
            'chatter': chatter, 'zigzag': zz}


def main():
    """Measure each set and log the medians."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--build', required=True, help='folder holding l2f_sim (same inputs as the policies)')
    parser.add_argument('--set', action='append', required=True, help='name=glob of train_mission.sh output files')
    parser.add_argument('--steps', default='1600000,1800000,2000000', help='checkpoint steps to use')
    parser.add_argument('--episodes', type=int, default=40, help='episodes per checkpoint')
    args = parser.parse_args()
    os.environ.update({'L2F_P_TAKEOFF': '0', 'L2F_P_LAND': '0', 'L2F_P_CIRCLE': '0', 'L2F_P_LINE': '0',
                       'L2F_P_EIGHT': '0', 'L2F_P_FAST_CIRCLE': '0', 'L2F_P_GUST': '0', 'L2F_P_P2P': '1',
                       'L2F_DR': '0', 'L2F_GLITCH_P': '0'})
    sys.path.insert(0, str(Path(args.build).resolve()))
    import l2f_sim
    info = l2f_sim.info()
    steps = [int(s) for s in args.steps.split(',')]
    log.info('point-to-point episodes, %d per checkpoint, checkpoints %s', args.episodes, steps)
    for spec in args.set:
        name, pattern = spec.split('=', 1)
        cks = checkpoints(sorted(glob.glob(pattern)), steps)
        res = []
        for i, ck in enumerate(cks):
            actor = load_actor_h5(Actor(info['obs_dim'], info['action_dim']), ck)
            sim = l2f_sim.Sim(1, 100 + i)
            sim.set_progress(1.0)
            res += [episode(sim, actor, info) for _ in range(args.episodes)]
        ok = [r for r in res if r['valid'] and not r['crashed']]
        med = lambda k: st.median(r[k] for r in ok if r[k] == r[k])
        whys = {}
        for r in res:
            if not r['valid']:
                whys[r['why']] = whys.get(r['why'], 0) + 1
        log.info('%-12s %d checkpoints, %d episodes, %d crashed, not measured %s: half-way %.2f s, overshoot %.0f%%, '
                 'settle %.2f s, peak speed %.2f m/s, chatter |da| %.4f, zig-zag cost %.5f',
                 name, len(cks), len(res), sum(r['crashed'] for r in res), whys or 0,
                 med('rise'), 100 * med('overshoot'), med('settle'), med('speed'), med('chatter'), med('zigzag'))


if __name__ == '__main__':
    main()
