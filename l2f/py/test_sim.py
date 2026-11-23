"""Check the l2f_sim Python module (plan A2) against the simulator itself.

1. Sim's step-by-step API reproduces reference_rollout (written directly against the
   C++ simulator, in rl_tools' runner call order) exactly: same observations, rewards
   and terminations for the same seed and actions, over several episodes.
2. The reward parts add up: sum_k w_k u_k equals the reward on every step.
3. A batch of environments steps with the right shapes.

Usage: python l2f/py/test_sim.py --build l2f/learning_to_fly/build_py_E/src [--steps 3000]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for run_log
import run_log  # noqa: E402

log = run_log.setup(__file__)


def actions_at(t, n_actions, hover):
    """The reference rollout's actions: clip(hover + 0.3 sin(0.05 t + k))."""
    a = np.array([hover + 0.3 * math.sin(0.05 * t + k) for k in range(n_actions)], dtype=np.float32)
    return np.clip(a, -1, 1)


def main(args):
    """Run the three checks; exit 1 if any fails."""
    sys.path.insert(0, str(Path(args.build).resolve()))
    import l2f_sim
    info = l2f_sim.info()
    log.info('module: obs %d, critic obs %d, actions %d, %d parts, step limit %d, mission %s, rpm obs %s',
             info['obs_dim'], info['obs_dim_privileged'], info['action_dim'], len(info['part_names']),
             info['step_limit'], info['mission'], info['rpm_obs'])
    ok = True

    ref = l2f_sim.reference_rollout(args.seed, args.steps)
    sim = l2f_sim.Sim(1, args.seed)
    sim.reset()
    episode_step, episodes, max_obs, max_rew, max_parts = 0, 1, 0.0, 0.0, 0.0
    w = sim.weights()
    term_mismatch = 0
    for t in range(args.steps):
        obs, _ = sim.observe()
        max_obs = max(max_obs, float(np.max(np.abs(obs[0] - ref['obs'][t]))))
        out = sim.step(ref['actions'][t][None, :])  # the reference's exact float32 actions
        max_rew = max(max_rew, abs(float(out['reward'][0]) - float(ref['reward'][t])))
        terms = w.astype(np.float64) * out['parts'][0].astype(np.float64)
        # relative to the size of the terms: float32 rounding of large costs (tumbling) is expected
        max_parts = max(max_parts, abs(terms.sum() - float(out['reward'][0])) / (np.abs(terms).sum() + 1))
        term_mismatch += bool(out['terminated'][0]) != bool(ref['terminated'][t])
        episode_step += 1
        if out['terminated'][0] or episode_step == info['step_limit']:
            sim.reset()
            episode_step = 0
            episodes += 1
    log.info('Sim vs reference, %d steps, %d episodes: max |obs diff| %.3g, max |reward diff| %.3g, '
             'termination mismatches %d', args.steps, episodes, max_obs, max_rew, term_mismatch)
    if max_obs != 0 or max_rew != 0 or term_mismatch:
        log.error('FAIL: the Sim API does not reproduce the reference rollout exactly')
        ok = False
    log.info('reward parts: max |sum_k w_k u_k - reward| / (sum_k |w_k u_k| + 1) = %.3g', max_parts)
    if max_parts > 1e-5:
        log.error('FAIL: the reward parts do not add up to the reward')
        ok = False

    batch = l2f_sim.Sim(4, args.seed + 1)
    batch.reset()
    obs, obs_p = batch.observe()
    out = batch.step(np.tile(actions_at(0, info['action_dim'], info['hover_action']), (4, 1)))
    shapes = (obs.shape, obs_p.shape, out['parts'].shape, out['reward'].shape)
    expected = ((4, info['obs_dim']), (4, info['obs_dim_privileged']), (4, len(info['part_names'])), (4,))
    log.info('batch of 4: shapes %s', shapes)
    if shapes != expected:
        log.error('FAIL: expected shapes %s', expected)
        ok = False
    log.info('all checks passed' if ok else 'checks FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--build', required=True, help='folder holding the built l2f_sim module')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--steps', type=int, default=3000)
    sys.exit(main(parser.parse_args()))
