"""Check the multi-objective TD3 (moo.py, plan A6) before any MOO comparison.

1. The 8 part groups add up to the scalar reward exactly (current weights, and with action
   rate and zig-zag both weighted).
2. With aggregator "sum", the TorchJD actor step equals the gradient of the summed value
   (one ordinary backward pass), and J^T w equals the step for upgrad.
3. --parts 1 (the MOO code path on the scalar reward) makes the same updates as td3.py: same
   initialization, same batches, critic and actor parameters agree after 20 updates.
4. The task batches hold only their task's transitions.

Usage: python l2f/py/test_moo.py --build l2f/learning_to_fly/build_py_E/src
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for run_log
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_log  # noqa: E402
from moo import MooTD3, group_parts  # noqa: E402
from td3 import TD3, ReplayBuffer, TD3Config  # noqa: E402

log = run_log.setup(__file__)


def fill(sim, info, steps, rng):
    """A replay buffer of `steps` random-action transitions."""
    buf = ReplayBuffer(steps, info['obs_dim'], info['obs_dim_privileged'], info['action_dim'], len(info['part_names']))
    done = np.array([True])
    n = 0
    for _ in range(steps):
        if done[0]:
            sim.reset(done)
            n = 0
        obs, obs_p = sim.observe()
        act = rng.uniform(-1, 1, (1, info['action_dim'])).astype(np.float32)
        out = sim.step(act)
        buf.add(obs, obs_p, act, out['parts'], out['next_obs'], out['next_obs_privileged'], out['terminated'], out['task'])
        n += 1
        done = out['terminated'] | (n == info['step_limit'])
    return buf


def params(m):
    """All parameters of a module as one vector."""
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()])


def main(args):
    """Run the four checks; exit 1 if any fails."""
    sys.path.insert(0, str(Path(args.build).resolve()))
    import l2f_sim
    info = l2f_sim.info()
    names = info['part_names']
    ok = True
    rng = np.random.default_rng(0)
    sim = l2f_sim.Sim(1, 0)
    sim.set_progress(1.0)
    buf = fill(sim, info, 20_000, rng)
    w = sim.weights()

    errs = []
    for weights in (w, np.where(np.array(names) == 'zigzag', 0.25, w)):
        U, W = group_parts(names, weights, 8)
        grouped = buf.parts[:buf.n] @ (U * W)
        scalar = buf.parts[:buf.n] @ np.asarray(weights, np.float32)
        errs.append(float(np.max(np.abs(grouped.sum(1) - scalar)) / max(float(np.max(np.abs(scalar))), 1e-12)))
    log.info('1. 8 groups vs scalar reward: max rel error %.2g (current weights), %.2g (rate and zig-zag both on); '
             '%d terminated transitions', errs[0], errs[1], int(buf.terminated[:buf.n].sum()))
    if max(errs) > 1e-5:
        log.error('FAIL: the groups do not add up to the reward')
        ok = False

    cfg = TD3Config(batch=256)
    torch.manual_seed(0)
    agent = MooTD3(cfg, info['obs_dim'], info['obs_dim_privileged'], info['action_dim'], agg='sum', log_every=1)
    R = agent.reward_matrix(names, w)
    obs, obs_p = buf.sample(np.random.default_rng(1), cfg.batch, R)[:2]
    agent.actor.zero_grad()
    (-(agent.critics[0](obs_p, agent.actor(obs)) * agent.factors).sum(1).mean()).backward()
    plain = torch.cat([p.grad.reshape(-1) for p in agent.actor.parameters()])  # before the step changes the actor
    agent.train_actor(buf, np.random.default_rng(1), R, 0)
    jd_step = torch.cat([p.grad.reshape(-1) for p in agent.actor.parameters()])  # the gradient the step used
    rel = float(torch.linalg.norm(jd_step - plain) / torch.linalg.norm(plain))
    up = MooTD3(cfg, info['obs_dim'], info['obs_dim_privileged'], info['action_dim'], agg='upgrad', log_every=1)
    up.train_actor(buf, np.random.default_rng(2), up.reward_matrix(names, w), 0)
    log.info('2. sum aggregator vs plain backward: rel diff %.2g; logged check (J^T w vs step) sum %.2g, upgrad %.2g; '
             'upgrad weights %s', rel, agent.rows[-1][-1], up.rows[-1][-1],
             np.round(up.rows[-1][2 + 8:2 + 16], 3).tolist())
    if rel > 1e-5 or agent.rows[-1][-1] > 1e-5 or up.rows[-1][-1] > 1e-5:
        log.error('FAIL: the TorchJD step does not match')
        ok = False

    torch.manual_seed(7)
    td3 = TD3(cfg, info['obs_dim'], info['obs_dim_privileged'], info['action_dim'])
    torch.manual_seed(7)
    moo1 = MooTD3(cfg, info['obs_dim'], info['obs_dim_privileged'], info['action_dim'], agg='sum', parts=1)
    same_init = torch.equal(params(td3.critics[0]), params(moo1.critics[0])) and torch.equal(params(td3.actor), params(moo1.actor))
    for agent_, seed in ((td3, 3), (moo1, 3)):
        r = np.random.default_rng(seed)
        torch.manual_seed(seed)
        for t in range(30_010, 30_010 + 200, 10):
            agent_.update(t, buf, r, names, w)
    d_critic = float(torch.max(torch.abs(params(td3.critics[0]) - params(moo1.critics[0]))))
    d_actor = float(torch.max(torch.abs(params(td3.actor) - params(moo1.actor))))
    log.info('3. parts=1 vs TD3: same init %s; after 20 critic / 10 actor updates max |diff| critic %.2g, actor %.2g',
             same_init, d_critic, d_actor)
    if not same_init or d_critic > 1e-5 or d_actor > 1e-5:
        log.error('FAIL: parts=1 does not reproduce TD3')
        ok = False

    present = sorted(set(buf.task[:buf.n].tolist()))
    pure = all((idx := buf.indices(rng, 256, task=t)) is None or bool(np.all(buf.task[idx] == t)) for t in range(9))
    log.info('4. tasks in the buffer %s; task batches pure: %s; absent task gives None: %s', present, pure,
             all(buf.indices(rng, 256, task=t) is None for t in range(9) if t not in present))
    if not pure:
        log.error('FAIL: a task batch holds other tasks')
        ok = False
    log.info('all checks passed' if ok else 'checks FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--build', required=True, help='folder holding l2f_sim')
    sys.exit(main(parser.parse_args()))
