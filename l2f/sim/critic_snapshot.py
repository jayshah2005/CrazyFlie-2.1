"""What a multi-output MOO critic learned: are its group values distinct, or copies of one?

Runs one training (train.py, unchanged) and, when the actor update reaches each --at step,
samples 1,024 replay states with its own RNG and records, for the critic's group values at the
policy's action: the correlation between the group values, the share of their variance along one
direction (top share; 1.0 = copies of one value), each group's fit to its own TD target (R^2), and
the same correlations for the groups' per-step rewards and targets. Training continues to the
last --at step and stops there; the snapshot does not change it (reruns reproduce the study logs).

Also (2026-11-27): hidden-layer saturation (FAST_TANH clips at +-3 with zero gradient), units
saturated on >= 99% of samples, effective feature rank (srank), the shared critic's head-weight
cosines, the cosine between the groups' action-gradients, and the twin critics' agreement on each
group's action-gradient (a reference-free check of slope quality).

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


def srank(h, delta=0.01):
    """Effective feature rank (Kumar et al. 2021): fewest singular values holding 1 - delta of their sum."""
    s = np.linalg.svd(h - h.mean(0), compute_uv=False)
    return int(np.searchsorted(np.cumsum(s) / s.sum(), 1 - delta) + 1)


def mlp_stats(net, x):
    """Hidden-layer saturation (|pre-activation| >= 3: FAST_TANH outputs +-1 and passes zero gradient),
    units saturated on >= 99% of samples, and the last hidden layer's effective rank."""
    from networks import fast_tanh
    norms = getattr(net, 'norms', None)  # --critic-layer-norm: LayerNorm before each activation
    z1 = net.layers[0](x) if norms is None else norms[0](net.layers[0](x))
    z2 = net.layers[1](fast_tanh(z1)) if norms is None else norms[1](net.layers[1](fast_tanh(z1)))
    h2 = fast_tanh(z2).numpy()
    sat = [float((z.abs() >= 3).float().mean()) for z in (z1, z2)]
    dead = [float(((z.abs() >= 3).float().mean(0) >= 0.99).float().mean()) for z in (z1, z2)]
    return {'saturated': sat, 'dead_units': dead, 'srank': srank(h2)}


def structure_stats(critic, obs_p, act):
    """Saturation and rank of the critic's hidden layers (shared critic, or each split network);
    cosine between the shared critic's output-layer rows (1 = heads read the features identically)."""
    from networks import Critic, SplitCritic
    x = torch.cat([obs_p, act], dim=-1)
    if isinstance(critic, SplitCritic):
        per = [mlp_stats(n, x) for n in critic.nets]
        return {'saturated': np.mean([p['saturated'] for p in per], 0).round(4).tolist(),
                'dead_units': np.mean([p['dead_units'] for p in per], 0).round(4).tolist(),
                'srank': [p['srank'] for p in per]}
    if isinstance(critic, Critic):
        s = mlp_stats(critic, x)
        w = critic.layers[-1].weight.detach()
        w = w / w.norm(dim=1, keepdim=True)
        s['head_weight_cos'] = (w @ w.T).numpy().round(3).tolist()
        return s
    return {}


def slope_stats(agent, obs, obs_p):
    """Per-group action-gradients f_k dQ_k/da at the policy's actions: mean cosine between groups
    (1 = the groups' slopes point the same way, the actor sees no conflict) and between the twin
    critics for the same group (a reference-free check of slope quality: twins trained on different
    batches should agree where the slope is well determined)."""
    a0 = agent.actor(obs).detach()

    def grads(c):
        a = a0.clone().requires_grad_(True)
        q = c(obs_p, a)
        return [torch.autograd.grad(q[:, k].sum(), a, retain_graph=True)[0] * agent.factors[k] for k in range(q.shape[1])]

    g0, g1 = grads(agent.critics[0]), grads(agent.critics[1])
    cos = lambda u, v: float(torch.nn.functional.cosine_similarity(u, v, dim=-1).mean())
    k = len(g0)
    between = [[round(cos(g0[i], g0[j]), 3) for j in range(k)] for i in range(k)]
    return {'slope_cos_between_groups': between, 'slope_twin_agreement': [round(cos(g0[i], g1[i]), 3) for i in range(k)],
            'slope_twin_agreement_total': round(cos(sum(g0), sum(g1)), 3),
            'slope_norm': [round(float(g.norm(dim=-1).mean()), 4) for g in g0]}


def mc_returns(agent, buf, reward_matrix, window=60_000, horizon=230):
    """Per-group discounted Monte Carlo returns of the behaviour policy along the latest `window`
    stored transitions (one environment, so the buffer holds whole trajectories in order). An episode
    ends at a termination or where the next state does not continue the trajectory (truncation or
    reset); states closer than `horizon` steps (gamma^230 ~ 0.1) to a truncation are dropped.
    Compares the critic's group values with these returns: does the critic's group-value correlation
    match the correlation of the true group returns? (2026-11-27: separate critics give +0.3..+0.75,
    the shared critic 0.93-0.99; which is right?)"""
    n = buf.n
    order = np.arange(n) if n < buf.cap else (buf.i + np.arange(buf.cap)) % buf.cap
    order = order[-window:]
    # dense future cost per group: the same returns without the terminal reward (crash penalty /
    # landing payout, the 'survival' part), i.e. what each group would still pay by flying on; the
    # typical size of these is what --crash-split's shares are meant to be proportional to
    rm_dense = reward_matrix.astype(np.float32).copy()
    names = getattr(agent, 'part_names', None)
    if names and 'survival' in names:
        rm_dense[names.index('survival'), :] = 0
    r_dense = buf.parts[order] @ rm_dense
    r = buf.parts[order] @ reward_matrix.astype(np.float32)
    term = buf.terminated[order] > 0
    cont = np.zeros(len(order), bool)
    cont[:-1] = np.abs(buf.next_obs_p[order[:-1]] - buf.obs_p[order[1:]]).max(1) < 1e-5
    cut = ~term & ~cont  # truncation, reset, or the window's end
    g, gd = np.zeros_like(r), np.zeros_like(r)
    to_cut = np.full(len(order), 10 ** 9)
    zero = np.zeros(r.shape[1], np.float32)
    nxt, nxd, dist = zero, zero, 10 ** 9
    for t in range(len(order) - 1, -1, -1):
        if term[t]:
            nxt, nxd, dist = zero, zero, 10 ** 9
            g[t], gd[t] = r[t], r_dense[t]
        else:
            if cut[t]:
                nxt, nxd, dist = zero, zero, 0
            g[t] = r[t] + agent.cfg.gamma * nxt
            gd[t] = r_dense[t] + agent.cfg.gamma * nxd
            dist += 1
        to_cut[t] = dist
        nxt, nxd = g[t], gd[t]
    valid = np.nonzero(to_cut > horizon)[0]
    if len(valid) < 200:
        return {'mc_states': int(len(valid))}
    pick = np.random.default_rng(9).choice(valid, min(2048, len(valid)), replace=False)
    idx = order[pick]
    with torch.no_grad():
        q = (agent.critics[0](torch.from_numpy(buf.obs_p[idx]), torch.from_numpy(buf.act[idx])) * agent.factors).numpy()
    G = g[pick]
    D = -gd[pick]  # dense future cost (positive)
    k = G.shape[1]
    iu = np.triu_indices(k, 1)
    return {'mc_states': int(len(pick)), 'mc_return_corr': corr(G).round(3).tolist(),
            'mc_corr_min': float(corr(G)[iu].min()), 'mc_corr_max': float(corr(G)[iu].max()),
            'value_corr_same_states': corr(q).round(3).tolist(),
            'value_vs_mc_corr': [round(float(np.corrcoef(q[:, j], G[:, j])[0, 1]), 3) for j in range(k)],
            'value_vs_mc_r2': [round(float(1 - np.var(G[:, j] - q[:, j]) / np.var(G[:, j])), 3) for j in range(k)],
            'mc_return_mean': G.mean(0).round(2).tolist(),
            'dense_future_cost_mean': D.mean(0).round(2).tolist(),
            'dense_future_cost_p95': np.percentile(D, 95, axis=0).round(2).tolist()}


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
    with torch.no_grad():
        structure = structure_stats(c, obs_p, agent.actor(obs))
    slopes = slope_stats(agent, obs, obs_p)
    mc = mc_returns(agent, buf, reward_matrix)
    r = reward.numpy()
    k = q_pol.shape[1]
    fit = [float(1 - np.var(y[:, i] - q_buf[:, i]) / np.var(y[:, i])) for i in range(k)]
    off = corr(q_pol)[np.triu_indices(k, 1)]
    return {'step': step, 'names': list(agent.names),
            'value_corr': corr(q_pol).round(3).tolist(), 'value_corr_min': float(off.min()), 'value_corr_max': float(off.max()),
            'value_top_share': top_share(q_pol), 'value_mean': q_pol.mean(0).round(3).tolist(),
            'value_std': q_pol.std(0).round(3).tolist(), 'reward_corr': corr(r).round(3).tolist(),
            'target_corr': corr(y).round(3).tolist(), 'fit_r2_per_group': [round(x, 3) for x in fit],
            'terminated_share': float(term.mean()), **structure, **slopes, **mc}


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
                log.info('step %d: saturated %s, dead units %s, srank %s, twin slope agreement %s (total %.2f)', step,
                         s.get('saturated'), s.get('dead_units'), s.get('srank'), s['slope_twin_agreement'],
                         s['slope_twin_agreement_total'])
                if 'mc_corr_min' in s:
                    log.info('step %d: Monte Carlo group-return correlation %.2f..%.2f (critic on the same states %s), '
                             'value vs return R2 %s', step, s['mc_corr_min'], s['mc_corr_max'],
                             s['value_corr_same_states'], s['value_vs_mc_r2'])
                    log.info('step %d: dense future cost per group: mean %s, 95th percentile %s', step,
                             s['dense_future_cost_mean'], s['dense_future_cost_p95'])
        return orig(self, buf, rng, reward_matrix, step)

    moo.MooTD3.train_actor = patched
    sys.argv = ['train.py'] + train_args
    train.main()


if __name__ == '__main__':
    main()
