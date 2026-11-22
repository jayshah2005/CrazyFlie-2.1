"""Check that the PyTorch TD3 updates are rl_tools' TD3 updates (plan A5).

Both start from the same weights (rl_tools' initialization, copied into PyTorch) and get the
same batches (real transitions from the simulator) and the same target-smoothing noise, in the
order of rl_tools' loop: per round both critics (own batch, own noise), then the critic
targets; every second round the actor and its target. The six networks must stay equal up to
float32 rounding (about 1e-5 of the largest weight after 1000 rounds). A difference here would
mean the training differs, not just its random draws.

Only the first ~1000 rounds are comparable: once the actor leaves saturation (FAST_TANH's
gradient is exactly 0 beyond |x| = 3, and Adam turns tiny gradients into full steps), rounding
differences grow by themselves. A PyTorch copy whose starting weights differ by 1e-7 drifts
from the unperturbed run in the same window and by as much as rl_tools does (2026-11-22).

--control (information only) adds a second PyTorch copy that gets every batch with its rows
shuffled: same loss, a different float32 summation order at every update. Its drift is a floor
for any two implementations, and the log shows the rl_tools difference as a multiple of it. A
bug in the update makes that multiple grow from round to round; a constant arithmetic
difference keeps it flat or lets it shrink as chaos takes over. 2026-11-22: rl_tools (Accelerate)
sits 10-30x above the floor on both E and I1 at 500 rounds, falling to ~4x (I1 actor) by 1000;
I1's actor is far more chaotic (the control alone grows ~160x from 500 to 1000 rounds, ~1.2x on
E), so I1 passes 1e-4 after ~500-1000 rounds without any update difference.

Usage: python l2f/py/test_td3.py --build l2f/learning_to_fly/build_py_E/src [--control]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for run_log
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_log  # noqa: E402
from td3 import TD3, TD3Config, polyak  # noqa: E402
from test_moo import fill  # noqa: E402

log = run_log.setup(__file__)
NETS = ['actor', 'actor_target', 'critic_1', 'critic_2', 'critic_target_1', 'critic_target_2']


def modules(agent):
    """The PyTorch networks under rl_tools' names."""
    return dict(zip(NETS, [agent.actor, agent.actor_target, agent.critics[0], agent.critics[1],
                           agent.critic_targets[0], agent.critic_targets[1]]))


def load(agent, weights):
    """Copy rl_tools' weights into the PyTorch networks."""
    with torch.no_grad():
        for name, m in modules(agent).items():
            for lin, (w, b) in zip(m.layers, weights[name]):
                lin.weight.copy_(torch.from_numpy(np.array(w)))
                lin.bias.copy_(torch.from_numpy(np.array(b)))


def weights_of(agent):
    """The PyTorch networks' parameters in ref.weights()'s layout (for the control comparison)."""
    return {name: [(lin.weight.detach().numpy().copy(), lin.bias.detach().numpy().copy()) for lin in m.layers]
            for name, m in modules(agent).items()}


def max_rel_diff(agent, weights):
    """Per network: max |PyTorch - rl_tools| / max |rl_tools| over all parameters."""
    out = {}
    for name, m in modules(agent).items():
        d = s = 0.0
        for lin, (w, b) in zip(m.layers, weights[name]):
            for t, r in ((lin.weight, w), (lin.bias, b)):
                r = np.array(r)
                d = max(d, float(np.max(np.abs(t.detach().numpy() - r))))
                s = max(s, float(np.max(np.abs(r))))
        out[name] = d / s
    return out


def main(args):
    """Run the comparison; exit 1 if the networks drift apart."""
    sys.path.insert(0, str(Path(args.build).resolve()))
    import l2f_sim
    info = l2f_sim.info()
    rng = np.random.default_rng(0)
    sim = l2f_sim.Sim(1, 0)
    sim.set_progress(1.0)
    buf = fill(sim, info, 20_000, rng)
    w = sim.weights()
    R = np.asarray(w, np.float32)[:, None]
    ref = l2f_sim.TD3Reference(args.seed)
    cfg = TD3Config()
    agent = TD3(cfg, info['obs_dim'], info['obs_dim_privileged'], info['action_dim'])
    weights = ref.weights()
    load(agent, weights)
    if abs(weights['gamma'] - cfg.gamma) > 1e-6:  # float32 0.99
        log.error('FAIL: rl_tools gamma %g, PyTorch %g', weights['gamma'], cfg.gamma)
        return 1
    # control: a second PyTorch copy from the same weights that gets every batch with its rows
    # shuffled. The batch-mean losses are the same, only the float32 summation order differs at
    # every update, as between two implementations (rl_tools on Accelerate vs PyTorch); its
    # drift is the floor that no exact port can stay below.
    twin, perm_rng = None, np.random.default_rng(args.seed + 1)  # own stream: the batches stay the same
    if args.control:
        twin = TD3(cfg, info['obs_dim'], info['obs_dim_privileged'], info['action_dim'])
        load(twin, weights)
    agents = [agent] + ([twin] if twin else [])
    worst = 0.0
    for r in range(1, args.rounds + 1):
        for i in range(2):
            batch = buf.sample(rng, cfg.batch, R)
            obs, obs_p, act, reward, next_obs, next_obs_p, term = batch
            noise = np.clip(rng.normal(0, cfg.target_noise_std, (cfg.batch, info['action_dim'])),
                            -cfg.target_noise_clip, cfg.target_noise_clip).astype(np.float32)
            ref.train_critic(i, obs.numpy(), obs_p.numpy(), act.numpy(), reward[:, 0].numpy(), next_obs.numpy(),
                             next_obs_p.numpy(), term.numpy() > 0.5, noise)
            agent.train_critic(i, batch, torch.from_numpy(noise))
            if twin:
                p = torch.from_numpy(perm_rng.permutation(cfg.batch))
                twin.train_critic(i, tuple(x[p] for x in batch), torch.from_numpy(noise)[p])
        if r % 2 == 0:
            obs, obs_p = buf.sample(rng, cfg.batch, R)[:2]
            ref.train_actor(obs.numpy(), obs_p.numpy())
            agent.actor_step(obs, obs_p)
            if twin:
                p = torch.from_numpy(perm_rng.permutation(cfg.batch))
                twin.actor_step(obs[p], obs_p[p])
        ref.update_critic_targets()
        for a in agents:
            for c, ct in zip(a.critics, a.critic_targets):
                polyak(c, ct, cfg.polyak)
        if r % 2 == 0:
            ref.update_actor_target()
            for a in agents:
                polyak(a.actor, a.actor_target, cfg.polyak)
        if r in (1, 10, 100) or r % 500 == 0 or r == args.rounds:
            d = max_rel_diff(agent, ref.weights())
            worst = max(worst, max(d.values()))
            log.info('after %4d rounds (%d critic, %d actor updates): max rel diff %s', r, 2 * r, r // 2,
                     ', '.join(f'{k} {v:.1e}' for k, v in d.items()))
            if twin:
                c = max_rel_diff(agent, weights_of(twin))
                log.info('           control (PyTorch vs PyTorch, batch rows shuffled): %s',
                         ', '.join(f'{k} {v:.1e}' for k, v in c.items()))
                log.info('           rl_tools difference / control: %s',
                         ', '.join(f'{k} {d[k] / max(c[k], 1e-12):.0f}x' for k in NETS))
    ok = worst < args.tol
    log.info('networks %s rl_tools (worst %.1e, tolerance %.0e)', 'match' if ok else 'DIFFER from', worst, args.tol)
    return 0 if ok else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--build', required=True, help='folder holding l2f_sim')
    parser.add_argument('--rounds', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tol', type=float, default=1e-4, help='largest allowed relative difference')
    parser.add_argument('--control', action='store_true',
                        help='also train a PyTorch copy on row-shuffled batches and log the rl_tools difference '
                             'as a multiple of its drift (information only)')
    sys.exit(main(parser.parse_args()))
