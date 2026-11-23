"""Reward check before training (l2f/docs/time_to_target_study.md, section 7.2): is flying worth
more than crashing in every episode type, with the weights training ends at?

TD3 maximizes the discounted return G_t = sum_j gamma^j r_(t+j) (gamma 0.99: about 1 s ahead).
With no survival bonus every r <= 0, and a crash ends the episode at -P (P = crash penalty, 200).
From any state where the costs still ahead add up to more than P, crashing is the better choice,
and training drifts toward it (the time-to-target study: a waypoint leg costs 250-500, every
method learned to crash).

The check rolls a policy (an actor .h5, deterministic actions) through the training episode mix
at full curriculum (progress 1) and the final reward weights (the weight curriculum's caps), and
reports per episode type:
  survived / crashed / landed   episode counts
  G0 survived                   median and 10th percentile of the return from the episode start
  crash better                  share of the survivors' states (first 2 s) with G_t < -P: states
                                where crashing at once would have paid more than flying on
  main costs                    the reward parts that make up most of the survivors' G0
and a sign check: every reward part except survival must be a cost (weighted value <= 0) on every
step that does not end the episode (the first study's power cost paid for idling).

Verdict per type: FAIL if crash-better states > 5%, WARN if any > 0% or the 10th percentile of
G0 is below -P/2, else OK. The result depends on the policy: a good mission policy (old3) shows
what flying well costs; it cannot show that a task the policy never learned is affordable.

Candidate rewards (--candidate, several allowed) are scored on the same flights: the policy's
flight does not depend on the reward, so each step's weighted parts and distance to the target d
(the position cost's d, before the step) are recorded once and the reward is recomputed. Roll out
WITHOUT L2F_POS_CAP so the recorded position part is the plain d^2 (the cap is reward-only).
  bonus=B        survival bonus B per step that does not end the episode
  crash=P        crash penalty P instead of the build's
  huber=C        position cost d^2 within C, C^2 + 2C(d - C) beyond, velocity cost 0 beyond C
                 (L2F_POS_CAP=C, the time-to-target study's runs 6 and later)
  progress=K[:C] position cost min(d, C)^2 (C default 0.2), velocity cost 0 beyond C, plus the
                 potential-based progress reward K (d_t - gamma d_t+1) and -K d_t on a crash
                 step (study section 5.6)
Several may be combined with '+', e.g. progress=20+bonus=0.5.

Usage (runtime options as in training, --env KEY=VALUE as train.py):
  python l2f/py/reward_check.py --build l2f/learning_to_fly/build_py_E_nb200/src \
      --actor l2f/policies/2026-11-22_old3_moosum3_s40_1800k/actor.h5 \
      --env L2F_OMEGA_CRASH=34.9 --env L2F_DELAY_MAX_MS=20 [--env L2F_P_WAYPOINT=0.2 ...] \
      [--candidate huber=0.2 --candidate progress=20 ...]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))  # repo root, for run_log
sys.path.insert(0, str(HERE))
import run_log  # noqa: E402
from export import load_actor_h5  # noqa: E402
from networks import Actor  # noqa: E402
from td3 import TD3Config  # noqa: E402

log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()

# train.py's runtime settings for training and the per-motor mismatch of variants A-E
TRAIN_ENV = {'L2F_MISSION_RAMP': '800000', 'L2F_GLITCH_P': '0.003', 'L2F_DR': '1', 'L2F_DR_SCALE': '0.3'}
MOTOR_GAIN, MOTOR_OFFSET = 0.05, 500
TASK_NAMES = ['free_flight', 'takeoff', 'landing', 'circle', 'line', 'eight', 'fast_circle', 'p2p', 'gust', 'waypoint']
STATE_WINDOW = 200    # states checked per episode: the first 2 s
TAIL = 300            # a survivor's G_t counts only with >= 300 steps left (gamma^300 = 0.05)
VERDICTS = ['OK', 'WARN', 'FAIL']


@dataclass
class Episode:
    """One recorded episode: rewards r_t, weighted parts (T, K), distance d_t before each step."""
    task: int
    reward: np.ndarray
    parts: np.ndarray
    dist: np.ndarray
    end: str  # 'crash', 'landed' or 'survived' (step limit)


def discounted(r: np.ndarray, gamma: float) -> np.ndarray:
    """G_t for every t of one episode (rewards r_0..r_T-1, nothing after the end)."""
    g = np.zeros(len(r) + 1)
    for t in range(len(r) - 1, -1, -1):
        g[t] = r[t] + gamma * g[t + 1]
    return g[:-1]


def final_weights(sim) -> np.ndarray:
    """Set and return the weights training ends at (td3.py weight_caps: the caps)."""
    sim.set_weights({k: cap for k, (_, cap) in TD3Config().weight_caps.items()})
    return np.asarray(sim.weights(), np.float64)


def make_candidate(spec: str, names: list[str], pos_scale: float, w_pos: float, gamma: float):
    """(Episode -> rewards, crash penalty or nan, progress K) for a candidate spec (see the module
    docstring). pos_scale x w_pos x cost is the weighted position cost (the recorded parts are weighted)."""
    opts = {}
    for item in spec.split('+'):
        key, _, val = item.partition('=')
        if key not in ('bonus', 'crash', 'huber', 'progress') or not val:
            raise SystemExit(f'--candidate {spec!r}: unknown item {item!r}')
        opts[key] = val
    i_pos, i_vel = names.index('position'), names.index('linear_velocity')

    def reward(e: Episode) -> np.ndarray:
        parts = e.parts.copy()
        crash_step = e.end == 'crash'
        d = e.dist
        extra = np.zeros(len(d))
        if 'huber' in opts or 'progress' in opts:
            c = float(opts['huber']) if 'huber' in opts else float((opts['progress'].split(':') + ['0.2'])[1])
            cost = np.where(d <= c, d ** 2, c ** 2 + 2 * c * (d - c)) if 'huber' in opts else np.minimum(d, c) ** 2
            parts[:, i_pos] = -pos_scale * w_pos * cost
            parts[:, i_vel] = np.where(d > c, 0.0, parts[:, i_vel])
        if 'progress' in opts:
            k = float(opts['progress'].split(':')[0])
            d_next = np.append(d[1:], d[-1])
            extra += k * (d - gamma * d_next)
            if crash_step:
                extra[-1] = -k * d[-1]  # no refund of the distance still to fly
        r = parts.sum(1) + extra
        if crash_step:
            r[-1] = -float(opts['crash']) if 'crash' in opts else e.reward[-1]
            if 'progress' in opts:
                r[-1] += extra[-1]
        elif e.end == 'landed':
            r[-1] = e.reward[-1]
        if 'bonus' in opts:
            r[:len(r) - (e.end != 'survived')] += float(opts['bonus'])
        return r
    k = float(opts['progress'].split(':')[0]) if 'progress' in opts else 0.0
    return reward, float(opts.get('crash', 'nan')), k


def analyze(episodes: list[Episode], reward_fn, P: float, k: float, gamma: float):
    """{task: (n_survived, n_crashed, n_landed, G0 median, G0 p10, crash-better %, verdict, G0 list)}.
    Crashing in state t is worth -(P + k d_t): the crash penalty plus, with the progress reward K = k,
    the distance still to fly."""
    out = {}
    for t in sorted({e.task for e in episodes}):
        eps = [e for e in episodes if e.task == t]
        surv = [e for e in eps if e.end != 'crash']
        n_crash = len(eps) - len(surv)
        n_land = sum(e.end == 'landed' for e in surv)
        if not surv:
            out[t] = (0, n_crash, 0, float('nan'), float('nan'), float('nan'), 'FAIL', [])
            continue
        g0, margins = [], []
        for e in surv:
            g = discounted(reward_fn(e), gamma)
            g0.append(g[0])
            last = len(g) if e.end == 'landed' else len(g) - TAIL  # a landing has no cut-off future
            n_states = max(0, min(STATE_WINDOW, last))
            margins.extend(g[:n_states] + P + k * e.dist[:n_states])  # < 0: crashing pays more
        g0, margins = np.array(g0), np.array(margins)
        better = 100 * np.mean(margins < 0) if len(margins) else float('nan')
        verdict = 'FAIL' if better > 5 else ('WARN' if better > 0 or np.percentile(g0, 10) < -P / 2 else 'OK')
        out[t] = (len(surv) - n_land, n_crash, n_land, np.median(g0), np.percentile(g0, 10), better, verdict, g0)
    return out


def main() -> None:
    """Roll the policy, then report per episode type, for the reward as built and each candidate."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--build', type=Path, required=True, help='build folder holding l2f_sim')
    ap.add_argument('--actor', type=Path, required=True, help='actor checkpoint (.h5) to roll out')
    ap.add_argument('--variant', default='E', help='I1 (no motor mismatch) or A-E, as train.py')
    ap.add_argument('--env', action='append', default=[], help='runtime option KEY=VALUE, as train.py')
    ap.add_argument('--candidate', action='append', default=[], help='candidate reward (see the header)')
    ap.add_argument('--episodes', type=int, default=1500, help='episodes in total (the training mix)')
    ap.add_argument('--envs', type=int, default=64, help='environments stepped together')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--crash-penalty', type=float, default=200, help='P, if no crash is seen (else measured)')
    ap.add_argument('--initial-weights', action='store_true',
                    help='keep the starting weights instead of the final ones (for comparison only)')
    a = ap.parse_args()

    env = dict(TRAIN_ENV)
    env['L2F_MOTOR_GAIN'] = str(MOTOR_GAIN if a.variant != 'I1' else 0)
    env['L2F_MOTOR_OFFSET'] = str(MOTOR_OFFSET if a.variant != 'I1' else 0)
    env.update(kv.split('=', 1) for kv in a.env)
    if a.candidate and float(env.get('L2F_PROGRESS_K', 0)) != 0:
        raise SystemExit('candidates need the plain position cost: roll out without L2F_PROGRESS_K '
                         '(score it as --candidate progress=K)')
    if a.candidate and float(env.get('L2F_POS_CAP', 0)) > 0:
        raise SystemExit('candidates need the plain position cost: roll out without L2F_POS_CAP '
                         '(score the cap as --candidate huber=C)')
    os.environ.update(env)  # read when the Sim is created
    sys.path.insert(0, str(a.build.resolve()))
    import l2f_sim
    info = l2f_sim.info()
    names = info['part_names']
    gamma = TD3Config().gamma
    torch.set_num_threads(1)
    actor = load_actor_h5(Actor(info['obs_dim'], info['action_dim']), a.actor).eval()

    sim = l2f_sim.Sim(a.envs, a.seed)
    sim.set_progress(1.0)
    w = np.asarray(sim.weights(), np.float64) if a.initial_weights else final_weights(sim)
    log.info('actor %s, %s weights: %s', a.actor, 'initial' if a.initial_weights else 'final',
             ', '.join(f'{n} {x:g}' for n, x in zip(names, w) if x))
    log.info('env %s', ' '.join(f'{k}={v}' for k, v in sorted(env.items())))

    # roll out and record
    n, limit = a.envs, info['step_limit']
    sim.reset(None)
    buf = [dict(r=[], p=[], d=[], task=-1) for _ in range(n)]
    episodes: list[Episode] = []
    sign_bad = np.zeros(len(names), int)
    sign_worst = np.zeros(len(names))
    penalties, scale_samples = [], []
    i_pos = names.index('position')
    k_built = float(w[names.index('progress')]) if 'progress' in names else 0.0  # progress reward K
    # parts that are rewards by design (not costs): survival and the progress reward
    rewards_ok = np.array([k == 0 or n == 'progress' for k, n in enumerate(names)])
    while len(episodes) < a.episodes:
        obs, _ = sim.observe()
        dist = np.linalg.norm(np.asarray(sim.states()['position'], np.float64), axis=1)  # before the step
        with torch.no_grad():
            act = actor(torch.from_numpy(np.asarray(obs, np.float32))).numpy()
        out = sim.step(act)
        term, landed = np.asarray(out['terminated']), np.asarray(out['landed'])
        wp = np.asarray(out['parts'], np.float64) * w
        for i in range(n):
            b = buf[i]
            if b['task'] < 0:
                b['task'] = int(out['task'][i])
            b['r'].append(float(out['reward'][i]))
            b['p'].append(wp[i])
            b['d'].append(dist[i])
            if not term[i]:  # costs only: every part but survival must be <= 0
                sign_bad += (wp[i] > 1e-6) & ~rewards_ok
                sign_worst = np.maximum(sign_worst, np.where(rewards_ok, 0, wp[i]))
                if dist[i] > 0.05 and w[i_pos] > 0:
                    scale_samples.append(-wp[i, i_pos] / (w[i_pos] * dist[i] ** 2))
            elif not landed[i]:
                penalties.append(-float(out['reward'][i]) - k_built * dist[i])  # the penalty without -K d
        done = term | np.array([len(b['r']) >= limit for b in buf])
        for i in np.flatnonzero(done):
            b = buf[i]
            end = 'landed' if landed[i] else ('crash' if term[i] else 'survived')
            episodes.append(Episode(b['task'], np.array(b['r']), np.array(b['p']), np.array(b['d']), end))
            buf[i] = dict(r=[], p=[], d=[], task=-1)
        if done.any():
            sim.reset(done)

    P = float(np.median(penalties)) if penalties else a.crash_penalty
    scale = float(np.median(scale_samples)) if scale_samples else float('nan')
    spread = float(np.percentile(scale_samples, 99) - np.percentile(scale_samples, 1)) if scale_samples else float('nan')
    log.info('%d episodes; crash penalty P = %g (%s); gamma %g; position cost = %.4g x w x d^2 (spread %.2g)',
             len(episodes), P, 'measured' if penalties else 'assumed', gamma, scale, spread)
    if a.candidate and not (spread < 1e-3 * scale):
        raise SystemExit('the recorded position part is not scale x d^2: cannot score candidates')

    bad = [f'{names[k]} ({sign_bad[k]} steps, up to +{sign_worst[k]:.3g})' for k in range(1, len(names)) if sign_bad[k]]
    if bad:
        log.error('SIGN CHECK FAILED: these parts paid a reward on ordinary steps: %s', '; '.join(bad))
    else:
        log.info('sign check OK: every part except survival and progress is a cost on every ordinary step')

    # the reward as built, in detail
    built = analyze(episodes, lambda e: e.reward, P, k_built, gamma)
    log.info('%-12s %5s %5s %5s | %-24s | %-12s | %s', 'type', 'surv', 'crash', 'land',
             'G0 survived: median/p10', 'crash better', 'main costs of survivors (median share of G0)')
    for t, (ns, nc, nl, med, p10, better, verdict, _) in built.items():
        name = TASK_NAMES[t] if 0 <= t < len(TASK_NAMES) else f'task {t}'
        if ns + nl == 0:
            log.error('%-12s %5d %5d %5d | no survivors: the policy cannot fly this type', name, 0, nc, 0)
            continue
        shares = []
        for e in episodes:
            if e.task == t and e.end != 'crash':
                disc = (gamma ** np.arange(len(e.reward)))[:, None] * e.parts
                if disc.sum() < 0:
                    shares.append(disc.sum(0) / disc.sum())
        share = np.median(np.array(shares), axis=0) if shares else np.zeros(len(names))
        top = ', '.join(f'{names[k]} {100 * share[k]:.0f}%' for k in np.argsort(-share)[:3] if share[k] > 0.05)
        log.log({'OK': 20, 'WARN': 30, 'FAIL': 40}[verdict], '%-12s %5d %5d %5d | %8.1f / %8.1f      | %5.1f%% %-4s | %s',
                name, ns, nc, nl, med, p10, better, verdict, top)
        if nc > 0.2 * (ns + nc + nl):
            log.warning('%-12s the policy crashes %.0f%% of these episodes: only the easy ones are left among the '
                        'survivors, so this type\'s numbers (and its candidate scores) look better than the reward '
                        'is; check it with a policy that flies it, or analytically', name, 100 * nc / (ns + nc + nl))
    worst = max((v[6] for v in built.values()), key=VERDICTS.index)
    log.info('verdict as built: %s (FAIL: crashing beats flying in >5%% of states of some type)', worst)

    # candidates: crash-better % (and median G0) per type, on the same flights
    if a.candidate:
        types = list(built)
        log.info('candidates (crash-better %% of survivor states / median G0 per type; same flights):')
        log.info('%-26s %s | verdict', 'candidate', ' '.join(f'{TASK_NAMES[t]:>17}' for t in types))
        for spec in ['as built'] + a.candidate:
            if spec == 'as built':
                res, p = built, P
            else:
                fn, p_new, k_new = make_candidate(spec, names, scale, float(w[i_pos]), gamma)
                p = P if np.isnan(p_new) else p_new
                res = analyze(episodes, fn, p, k_new, gamma)
            cells = ' '.join(f'{res[t][5]:6.1f}% {res[t][3]:8.1f}' if not np.isnan(res[t][5]) else f'{"no survivors":>17}'
                             for t in types)
            v = max((res[t][6] for t in types), key=VERDICTS.index)
            log.log({'OK': 20, 'WARN': 30, 'FAIL': 40}[v], '%-26s %s | %s', spec, cells, v)


if __name__ == '__main__':
    main()
