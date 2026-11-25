"""What a multi-output MOO critic learned: are its group values distinct, or copies of one?

Runs one training (train.py, unchanged) and, when the actor update reaches each --at step,
samples 1,024 replay states with its own RNG and records, for the critic's group values at the
policy's action: the correlation between the group values, the share of their variance along one
direction (top share; 1.0 = copies of one value), each group's fit to its own TD target (R^2), and
the same correlations for the groups' per-step rewards and targets. Training continues to the
last --at step and stops there; the snapshot does not change it (reruns reproduce the study logs).

Measured 2026-11-24 (seed 70, time 0.5 recipe): shared head-scaled critic 0.99-1.00 correlation at
200k (collapsed); split critic +0.29..+0.68 at 1.2M with R^2 0.98-1.00.

Usage: python l2f/sim/critic_snapshot.py --out snap.json --at 200000 1200000 -- \
         --seed 70 --moo parts --parts 4 --total 1 --agg sum --split-critic --env L2F_W_TIME=0.5 ...
(everything after -- goes to train.py; --build defaults to the E build, --steps to the last --at + 101)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'l2f' / 'py'))
sys.path.insert(0, str(ROOT))
import run_log  # noqa: E402

log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()


def corr(x):
    x = x - x.mean(0)
    s = x.std(0) + 1e-12
    return (x.T @ x / len(x)) / np.outer(s, s)


def top_share(x):
    ev = np.linalg.eigvalsh(np.cov((x - x.mean(0)).T))
    return float(ev[-1] / ev.sum())


def snapshot(agent, buf, reward_matrix, step):
    """Group values, rewards and TD targets on 1,024 replay states."""
    rng = np.random.default_rng(7)
    obs, obs_p, act, reward, next_obs, next_obs_p, term = buf.sample(rng, 1024, reward_matrix)
    with torch.no_grad():
        c = agent.critics[0]
        q_pol = (c(obs_p, agent.actor(obs)) * agent.factors).numpy()
        q_buf = (c(obs_p, act) * agent.factors).numpy()
        next_act = agent.actor_target(next_obs).clamp(-1, 1)
        q1 = agent.critic_targets[0](next_obs_p, next_act)
        q2 = agent.critic_targets[1](next_obs_p, next_act)
        pick = ((q1 * agent.factors).sum(1) < (q2 * agent.factors).sum(1))[:, None]
        y = (reward + agent.cfg.gamma * (1 - term[:, None]) * torch.where(pick, q1, q2)).numpy()
    r = reward.numpy()
    k = q_pol.shape[1]
    fit = [float(1 - np.var(y[:, i] - q_buf[:, i]) / np.var(y[:, i])) for i in range(k)]
    off = corr(q_pol)[np.triu_indices(k, 1)]
    return {'step': step, 'names': list(agent.names),
            'value_corr': corr(q_pol).round(3).tolist(), 'value_corr_min': float(off.min()), 'value_corr_max': float(off.max()),
            'value_top_share': top_share(q_pol), 'value_mean': q_pol.mean(0).round(3).tolist(),
            'value_std': q_pol.std(0).round(3).tolist(), 'reward_corr': corr(r).round(3).tolist(),
            'target_corr': corr(y).round(3).tolist(), 'fit_r2_per_group': [round(x, 3) for x in fit],
            'terminated_share': float(term.mean())}


def main():
    argv = sys.argv[1:]
    split = argv.index('--') if '--' in argv else len(argv)
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, required=True, help='JSON file for the snapshots')
    ap.add_argument('--at', type=int, nargs='+', default=[200_000, 1_200_000], help='training steps to snapshot')
    a = ap.parse_args(argv[:split])
    train_args = argv[split + 1:]
    if '--build' not in train_args:
        train_args += ['--build', str(ROOT / 'l2f/learning_to_fly/build_py_E_nb200/src')]
    if '--steps' not in train_args:
        train_args += ['--steps', str(max(a.at) + 101)]
    import moo
    import train
    snaps, done = {}, set()
    orig = moo.MooTD3.train_actor

    def patched(self, buf, rng, reward_matrix, step):
        for t in a.at:
            if step >= t and t not in done:
                done.add(t)
                s = snapshot(self, buf, reward_matrix, step)
                snaps[str(step)] = s
                a.out.write_text(json.dumps(snaps, indent=1))
                log.info('step %d: group-value correlation %.2f..%.2f, top share %.3f, fit R2 %s', step,
                         s['value_corr_min'], s['value_corr_max'], s['value_top_share'], s['fit_r2_per_group'])
        return orig(self, buf, rng, reward_matrix, step)

    moo.MooTD3.train_actor = patched
    sys.argv = ['train.py'] + train_args
    train.main()


if __name__ == '__main__':
    main()
