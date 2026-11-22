"""Closed-loop check of the sensor path (L2F_SENSOR_SIM, plan E3 and E5 speed).

A trained actor flies the full mission mix in the simulator. On a sensor-path build it sees
the flown estimator's output; the script reports how far that estimate is from the truth
(position, velocity, tilt, yaw, gyro sample) and how episodes end. On an old-path build it
reports how episodes end, for comparison. Both report simulator steps per second.

  --ideal     sensors without noise, bias, gaps, geometry error or drop-outs (config/sensors.yaml
              with those set to 0); with --no-force the estimate must then nearly equal the
              truth, which checks the wiring (frames, timing, units)
  --no-force  no random external force (L2F_FORCE_SCALE=0) and no domain randomization. The
              recipe's force (sigma m g / 20 per axis) is read by the flown filter as tilt, so
              with it the tilt error is ~2-3 deg even with ideal sensors (2026-11-22)

The priors in config/sensors.yaml are not measured yet: without --ideal these numbers show
what the priors do, not what the drone does (Step 5 checks that against the survey).

Usage: python l2f/py/test_sensor_sim.py --build l2f/learning_to_fly/build_py_E_ss/src --checkpoint <actor.h5> [--ideal --no-force]
       python l2f/py/test_sensor_sim.py --build l2f/learning_to_fly/build_py_E/src --checkpoint <actor.h5>
"""
import argparse
import math
import os
import re
import sys
import tempfile
import time
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
TASKS = ['free flight', 'take-off', 'landing', 'circle', 'line', 'eight', 'fast circle', 'p2p', 'gust']
# sensor-error keys set to 0 by --ideal (in every section of config/sensors.yaml)
IDEAL_ZERO = ['gyro_bias', 'gyro_noise_avg', 'gyro_noise_sample', 'gyro_noise_sample_x', 'gyro_noise_sample_y',
              'gyro_noise_sample_z', 'acc_bias', 'acc_noise_avg', 'rest_gap', 'bias',
              'noise', 'below_min_bias', 'below_min_noise', 'angle_noise', 'geometry_error_position',
              'geometry_error_rotation', 'dropout_rate']


def ideal_config(folder):
    """config/sensors.yaml with every sensor error set to 0, written into folder; its path."""
    lines = []
    for line in (ROOT / 'config' / 'sensors.yaml').read_text().splitlines():
        m = re.match(r'^(\s+)(\w+):', line)
        if m and m.group(2) in IDEAL_ZERO:
            line = f'{m.group(1)}{m.group(2)}: 0'
        elif m and m.group(2) == 'system':  # keep the Lighthouse file, as an absolute path
            line = f"{m.group(1)}system: {ROOT / line.split(':', 1)[1].split('#')[0].strip()}"
        lines.append(line)
    path = Path(folder) / 'sensors_ideal.yaml'
    path.write_text('\n'.join(lines) + '\n')
    return path


def rot(q):
    """Rotation matrix (body to world) of a quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def tilt_yaw_error(q_true, q_est):
    """(angle between the body z axes, yaw difference) in degrees."""
    rt, re_ = rot(q_true), rot(q_est)
    tilt = math.degrees(math.acos(float(np.clip(rt[:, 2] @ re_[:, 2], -1, 1))))
    yaw = math.degrees(math.atan2(re_[1, 0], re_[0, 0]) - math.atan2(rt[1, 0], rt[0, 0]))
    return tilt, abs((yaw + 180) % 360 - 180)


def main(args):
    """Fly the episodes, report estimate errors, episode outcomes and speed."""
    tmp = tempfile.TemporaryDirectory()
    if args.ideal:
        os.environ['L2F_SENSORS'] = str(ideal_config(tmp.name))
    if args.no_force:
        os.environ['L2F_FORCE_SCALE'] = '0'
        os.environ['L2F_DR'] = '0'
    sys.path.insert(0, str(Path(args.build).resolve()))
    import l2f_sim
    info = l2f_sim.info()
    ss = info.get('sensor_sim', False)  # modules built before the key existed are old-path
    actor = load_actor_h5(Actor(info['obs_dim'], info['action_dim']), args.checkpoint)
    sim = l2f_sim.Sim(1, args.seed)
    sim.set_progress(1.0)
    err = {'position': [], 'velocity': [], 'tilt': [], 'yaw': [], 'gyro': []}
    per_task = {}
    steps_total, t_sim = 0, 0.0
    for _ in range(args.episodes):
        sim.reset(np.array([True]))
        task = int(sim.states()['task'][0])
        n, early = 0, False
        while n < info['step_limit']:
            obs, _ = sim.observe()
            with torch.no_grad():
                act = actor(torch.from_numpy(obs)).numpy()
            t0 = time.perf_counter()
            out = sim.step(act)
            t_sim += time.perf_counter() - t0
            n += 1
            if ss and n > args.skip:
                s = sim.states()
                err['position'].append(np.linalg.norm(s['est_position'][0] - s['position'][0]))
                err['velocity'].append(np.linalg.norm(s['est_velocity'][0] - s['linear_velocity'][0]))
                tilt, yaw = tilt_yaw_error(s['orientation'][0], s['est_orientation'][0])
                err['tilt'].append(tilt)
                err['yaw'].append(yaw)
                err['gyro'].append(np.linalg.norm(s['gyro_obs'][0] - s['angular_velocity'][0]))
            if out['terminated'][0]:
                early = not bool(out['landed'][0])
                break
        steps_total += n
        k = per_task.setdefault(task, [0, 0, 0])
        k[0] += 1
        k[1] += early
        k[2] += n
    log.info('%s build%s%s, %d episodes, checkpoint %s', 'sensor-path' if ss else 'old-path',
             ', ideal sensors' if args.ideal else '', ', no random force or randomization' if args.no_force else '',
             args.episodes, Path(args.checkpoint).name)
    for task, (count, early, steps) in sorted(per_task.items()):
        log.info('  %-12s %3d episodes, %3d ended early (crash), mean length %.0f steps', TASKS[task], count, early,
                 steps / count)
    total = sum(v[1] for v in per_task.values())
    log.info('  all          %3d episodes, %.1f%% ended early', args.episodes, 100 * total / args.episodes)
    log.info('simulator: %.0f steps/s (%.1f us per step)', steps_total / t_sim, 1e6 * t_sim / steps_total)
    ok = True
    if ss:
        e = {k: np.array(v) for k, v in err.items()}
        pct = lambda a, q: float(np.percentile(a, q))
        log.info('estimate - truth over %d steps (after step %d of each episode), median / 90%%: position %.1f / %.1f mm, '
                 'velocity %.0f / %.0f mm/s, tilt %.2f / %.2f deg, yaw %.2f / %.2f deg, gyro sample %.3f / %.3f rad/s',
                 len(e['position']), args.skip, 1e3 * pct(e['position'], 50), 1e3 * pct(e['position'], 90),
                 1e3 * pct(e['velocity'], 50), 1e3 * pct(e['velocity'], 90), pct(e['tilt'], 50), pct(e['tilt'], 90),
                 pct(e['yaw'], 50), pct(e['yaw'], 90), pct(e['gyro'], 50), pct(e['gyro'], 90))
        finite = all(np.all(np.isfinite(v)) for v in e.values())
        if args.ideal and args.no_force:
            # wiring check: ideal sensors, nothing the filter's model lacks
            ok = finite and pct(e['position'], 50) < 0.003 and pct(e['tilt'], 50) < 0.5 and pct(e['yaw'], 50) < 0.2
            limits = 'position median < 3 mm, tilt median < 0.5 deg, yaw median < 0.2 deg'
        else:
            ok = finite and pct(e['position'], 50) < 0.02 and pct(e['tilt'], 50) < 5
            limits = 'position median < 20 mm, tilt median < 5 deg (sanity)'
        log.info('estimate %s (%s)', 'within limits' if ok else 'FAILS the limits', limits)
    tmp.cleanup()
    return 0 if ok else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--build', required=True, help='folder holding l2f_sim')
    parser.add_argument('--checkpoint', required=True, help='actor .h5 of the same variant')
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip', type=int, default=20, help='first steps of each episode left out of the errors')
    parser.add_argument('--ideal', action='store_true', help='sensors without errors')
    parser.add_argument('--no-force', action='store_true', help='no random force, no domain randomization')
    sys.exit(main(parser.parse_args()))
