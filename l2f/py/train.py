"""Train the mission policy with PyTorch TD3 on the C++ simulator (plan A4).

The simulator is the l2f_sim module of a build (same -D flags as l2f/sim/train_mission.sh,
built with -DL2F_BUILD_PYTHON=ON); this script sets the same runtime settings as
train_mission.sh (curriculum ramp, Lighthouse jumps, randomization, per-motor mismatch for
variants A-E) before creating it. Checkpoints go to
l2f/learning_to_fly/checkpoints/multirotor_td3/<time>_py_<tag>/actor_<step>.h5, in rl_tools'
format. With --evaluate, the late checkpoints are scored by evaluate_robustness exactly as
train_mission.sh scores them, in the same table format (l2f/sim/summarize_checkpoints.py reads it).
--moo parts|tasks trains the multi-objective version (moo.py, TorchJD aggregator --agg); its
per-objective gradient log goes to logs/train_moo_<tag>_<pid>_<time>.csv.

Usage: python l2f/py/train.py --build l2f/learning_to_fly/build_py_E/src --variant E --seed 0 --evaluate
       python l2f/py/train.py ... --moo parts --agg upgrad [--parts 1] [--total 1] [--norm]
       python l2f/py/train.py ... --moo tasks --tasks 9 --agg task_share
       extra runtime options as KEY=VALUE, e.g. --env L2F_LANDING_CUT=1
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))  # run_log
sys.path.insert(0, str(HERE))
import run_log  # noqa: E402
from export import save_actor_h5  # noqa: E402
from moo import AGGREGATORS, MooTD3  # noqa: E402
from td3 import TD3, ReplayBuffer, TD3Config  # noqa: E402

log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()
BASE_ENV = dict(os.environ)  # evaluation starts from this, without the training settings (as train_mission.sh)
L2F = ROOT / 'l2f' / 'learning_to_fly'
MOTOR_GAIN, MOTOR_OFFSET = 0.05, 500  # per-motor mismatch of variants A-E (train_mission.sh)
# train_mission.sh's runtime settings for training
TRAIN_ENV = {'L2F_MISSION_RAMP': '800000', 'L2F_GLITCH_P': '0.003', 'L2F_DR': '1', 'L2F_DR_SCALE': '0.3'}
# train_mission.sh's ev(): 0.75 m, 0.6 m circle in 10 s, 6 ms delay, ground effect, no outer correction
EVAL_ENV = {'L2F_MISSION_H': '0.75', 'L2F_MISSION_R': '0.6', 'L2F_MISSION_P': '10', 'L2F_MISSION_V_DOWN': '0.25',
            'L2F_EVAL_GE': '1', 'L2F_EVAL_NOISE': '1', 'L2F_DR_SCALE': '0.5', 'L2F_EVAL_DELAY_MS': '6',
            'L2F_EVAL_KI': '0', 'L2F_MOTOR_GAIN': str(MOTOR_GAIN), 'L2F_MOTOR_OFFSET': str(MOTOR_OFFSET)}
# --tasks 9 / --hard (train_mission.sh HARD=1): every mission episode type 10%, free flight 20%
HARD_ENV = {f'L2F_P_{k}': '0.1' for k in ('TAKEOFF', 'LAND', 'CIRCLE', 'LINE', 'EIGHT', 'FAST_CIRCLE', 'P2P', 'GUST')}


def make_agent(args, cfg, info, weights=None):
    """TD3, or the multi-objective version for --moo."""
    dims = (info['obs_dim'], info['obs_dim_privileged'], info['action_dim'])
    if not args.moo:
        return TD3(cfg, *dims)
    objectives = None
    if args.parts == 0 and weights is not None:
        # ungrouped: every reward part that is weighted now or by the weight curriculum later
        objectives = [n for n, w in zip(info['part_names'], weights) if w != 0 or n in cfg.weight_caps]
    return MooTD3(cfg, *dims, agg=args.agg, parts=args.parts, tasks=args.tasks if args.moo == 'tasks' else 0,
                  total=args.total, norm=args.norm, objectives=objectives, grouping=args.grouping)


def run_tag(args):
    """Run-folder tag: variant, MOO options, seed."""
    tag = f'py_{args.variant}'
    if args.moo == 'parts':
        tag += f'_moo{args.parts}{"r" if args.grouping == "regroup" else ""}_{args.agg}' + (f'_tot{args.total:g}' if args.total is not None else '') + \
            ('_norm' if args.norm else '')
    elif args.moo == 'tasks':
        tag += f'_tasks{args.tasks}_{args.agg}'
    if args.hard:
        tag += '_hard'
    return tag + f'_s{args.seed}'


def train(args, extra_env):
    """Run the TD3 loop (td3.py) and write checkpoints; returns the run folder."""
    env = dict(TRAIN_ENV)
    mismatch = args.variant != 'I1'
    env['L2F_MOTOR_GAIN'] = str(MOTOR_GAIN if mismatch else 0)
    env['L2F_MOTOR_OFFSET'] = str(MOTOR_OFFSET if mismatch else 0)
    if args.hard or (args.moo == 'tasks' and args.tasks == 9):
        env.update(HARD_ENV)
    env.update(extra_env)
    os.environ.update(env)  # the simulator reads these when a Sim is created
    sys.path.insert(0, str(Path(args.build).resolve()))
    import l2f_sim
    info = l2f_sim.info()
    cfg = TD3Config(steps=args.steps, mission_ramp=int(env['L2F_MISSION_RAMP']))
    waypoint_ramp = int(float(env.get('L2F_WAYPOINT_RAMP', 0)))
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)  # 64-unit networks: threads cost more than they save
    rng = np.random.default_rng(args.seed)
    sim = l2f_sim.Sim(1, args.seed)
    agent = make_agent(args, cfg, info, sim.weights())
    buf = ReplayBuffer(cfg.replay_cap, info['obs_dim'], info['obs_dim_privileged'], info['action_dim'],
                       len(info['part_names']))
    names = info['part_names']
    weight_index = {k: names.index(k) for k in cfg.weight_caps}
    # off only in the untuned-reward build (REWARD=uniform); modules built before the key existed are tuned
    weight_curriculum = info.get('weight_curriculum', True)
    tag = run_tag(args) + ('' if weight_curriculum else '_uniform') + ('_ss' if info.get('sensor_sim') else '')
    run = L2F / 'checkpoints' / 'multirotor_td3' / (STARTED_AT.strftime('%Y_%m_%d_%H_%M_%S') + f'_{tag}_{os.getpid()}')
    moo_log = run_log.csv_path(f'moo_{tag}_{os.getpid()}', script=__file__, when=STARTED_AT) if args.moo else None
    run.mkdir(parents=True, exist_ok=True)
    log.info('run %s: variant %s, seed %d, %d steps, obs %d / critic %d, %s, env %s', run.name, args.variant,
             args.seed, cfg.steps, info['obs_dim'], info['obs_dim_privileged'],
             f'MOO {args.moo} ({", ".join(agent.names)}), aggregator {args.agg}' if args.moo else 'TD3',
             ' '.join(f'{k}={v}' for k, v in sorted(env.items())))
    episode_step, episode_return, returns = 0, 0.0, []
    done = np.array([True])
    t0 = time.time()
    for t in range(cfg.steps):
        # curriculum and checkpoint before the step, as in training.h
        if t % 10_000 == 0:
            sim.set_progress(min(1.0, t / cfg.mission_ramp) if cfg.mission_ramp > 0 else 1.0)
            if waypoint_ramp > 0:  # B: the waypoint jumps grow on their own (slower) schedule
                sim.set_waypoint_progress(min(1.0, t / waypoint_ramp))
        if weight_curriculum and t != 0 and t % 100_000 == 0 and t != cfg.steps - 1:
            w = sim.weights()
            sim.set_weights({k: min(cap, float(w[weight_index[k]]) * f) for k, (f, cap) in cfg.weight_caps.items()})
        if t % 100_000 == 0 and t >= 500_000:
            agent.decay_noise()
        if t % cfg.checkpoint_interval == 0:
            save_actor_h5(agent.actor, run / f'actor_{t:015d}.h5')
            if moo_log and agent.rows:
                run_log.write_csv(moo_log, agent.log_fields(), agent.rows, script=__file__,
                                  generated_at=datetime.now().astimezone())
        if t % 100_000 == 0 and t:
            log.info('step %d: mean training return %.1f over %d episodes (%.0f s)', t,
                     np.mean(returns[-200:]) if returns else float('nan'), len(returns), time.time() - t0)
        if done[0]:
            sim.reset(done)
            if t:
                returns.append(episode_return)
            episode_step, episode_return = 0, 0.0
        obs, obs_p = sim.observe()
        act = agent.act(obs, rng)
        out = sim.step(act)
        buf.add(obs, obs_p, act, out['parts'], out['next_obs'], out['next_obs_privileged'], out['terminated'],
                out['task'])
        episode_step += 1
        episode_return += float(out['reward'][0])
        done = out['terminated'] | (episode_step == cfg.step_limit)
        agent.update(t, buf, rng, names, sim.weights())
    log.info('Training took: %.1f s', time.time() - t0)
    if moo_log and agent.rows:
        run_log.write_csv(moo_log, agent.log_fields(), agent.rows, script=__file__,
                          generated_at=datetime.now().astimezone())
        log.info('MOO gradient log: %s', moo_log)
    return run, env


def first_number(text, prefix):
    """The number right after `prefix` in text (as train_mission.sh's first_number)."""
    i = text.find(prefix)
    if i < 0:
        return ''
    j = i + len(prefix)
    while j < len(text) and text[j] == ' ':
        j += 1
    k = j
    while k < len(text) and (text[k].isdigit() or text[k] == '.'):
        k += 1
    return text[j:k]


def evaluate(run, evaluator, episodes, extra_env):
    """train_mission.sh's checkpoint table: clean / randomized / 1% jumps per checkpoint."""
    def ev(ck, dr, glitch):
        # later entries win, so --env can override EVAL_ENV (e.g. L2F_EVAL_DELAY_MS)
        env = {**BASE_ENV, **EVAL_ENV, **extra_env, 'L2F_EVAL_DR': str(dr), 'L2F_GLITCH_P': str(glitch)}
        out = subprocess.run([str(evaluator), str(ck), str(episodes), 'mission'], env=env,
                             capture_output=True, text=True).stdout
        circle = ''
        for line in out.splitlines():
            if line.strip().startswith('circle'):
                circle = first_number(line, 'tracking error mean')
        return f"crash {first_number(out, 'crashed '):>2} circle {circle:>5} cm jitter {first_number(out, 'metric) ')}"
    log.info('checkpoint   clean                             | randomized                        | jumps 1%')
    for ck in sorted(run.glob('actor_*.h5'))[-11:]:
        step = int(ck.stem.split('_')[1])
        log.info('%d  %s | %s | %s', step, ev(ck, 0, 0), ev(ck, 1, 0), ev(ck, 0, 0.01))


def main():
    """Parse the options, train, and evaluate."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--build', required=True, help='build folder holding l2f_sim (and evaluate_robustness)')
    parser.add_argument('--variant', default='E', help='I1 (no motor mismatch) or A-E (mismatch), as train_mission.sh')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=2_000_001)
    parser.add_argument('--env', action='append', default=[], help='extra runtime option KEY=VALUE')
    parser.add_argument('--evaluate', action='store_true', help='score the late checkpoints afterwards')
    parser.add_argument('--evaluate-run', type=Path, help='only score the checkpoints of this run folder')
    parser.add_argument('--moo', choices=['parts', 'tasks'], help='multi-objective TD3 (moo.py): framing A or B')
    parser.add_argument('--agg', default='sum', choices=AGGREGATORS, help='TorchJD aggregator (--moo)')
    parser.add_argument('--parts', type=int, default=8, choices=[0, 1, 3, 4, 5, 8], help='--moo parts: 8 objectives, 3 groups '
                        '(tracking, smoothness, safety), 4 groups (time, precision, efficiency, safety), 0 (ungrouped: one '
                        'objective per weighted reward part) or 1 (all merged: the MOO code path on the scalar '
                        'reward, must match TD3)')
    parser.add_argument('--grouping', default='default', choices=['default', 'regroup'],
                        help='--moo parts: regroup = crash, progress, effort, limits (--parts 4) + energy (--parts 5)')
    parser.add_argument('--tasks', type=int, default=5, choices=[5, 9], help='--moo tasks: episode types')
    parser.add_argument('--total', type=float, help='--moo parts: critic loss on the total error too')
    parser.add_argument('--norm', action='store_true', help='--moo parts: outputs scaled to O(1)')
    parser.add_argument('--hard', action='store_true', help='the harder episode mix of --tasks 9 (HARD=1)')
    parser.add_argument('--episodes', type=int, default=15, help='evaluation episodes per condition')
    args = parser.parse_args()
    extra = dict(kv.split('=', 1) for kv in args.env)
    if args.evaluate_run:
        evaluate(args.evaluate_run, Path(args.build) / 'evaluate_robustness', args.episodes, extra)
        return
    run, _ = train(args, extra)
    if args.evaluate:
        evaluate(run, Path(args.build) / 'evaluate_robustness', args.episodes, extra)


if __name__ == '__main__':
    main()
