"""Real-flight metrics in the simulator test's terms, from rl_trigger.py mission logs.

For each flight log (logs/rl_trigger_flight_*.csv, --mode mission, circle or --waypoints):
  - tracking error per phase (mean and max distance to the target while the policy flies), as
    evaluate_robustness prints it; "circle" is the sim test's circle number;
  - hover jitter (the sim's paper metric): per motor, the standard deviation of the policy's
    command u = m / 65535 over the hover phases, averaged over the 4 motors;
  - waypoint legs (phases 'waypoint' and 'back'): arrival within 10 cm and 5 cm (time after the
    target moved), overshoot (largest distance past the target along the leg), final error (mean
    over the leg's last 0.5 s), peak speed; as the sim's waypoint test reports them;
  - landing: distance from the take-off point when the motors stopped; crashed / aborted flag
    from the phases the log reached.
Writes one row per flight (and per leg with --legs) to a CSV under logs/.

Usage: python l2f/sim/flight_report.py --label svdpga logs/rl_trigger_flight_<time>.csv [...]
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_log  # noqa: E402

log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()
PHASES = ['takeoff', 'hover', 'move', 'circle', 'waypoint', 'back', 'hover2', 'landing']


def rows_of(path: Path) -> list[dict]:
    """Flight log rows with numbers parsed (empty cells None)."""
    with path.open() as f:
        out = []
        for r in csv.DictReader(line for line in f if not line.startswith('#')):
            out.append({k: (r[k] if k == 'phase' else (float(r[k]) if r[k] not in ('', None) else None)) for k in r})
    return out


def leg_metrics(leg: list[dict]) -> dict:
    """Arrival, overshoot, final error and peak speed of one waypoint leg. The leg's target is the
    median of the rebuilt target over the leg (its first rows can still carry the old error)."""
    t0 = leg[0]['drone_ms']
    tgt = [st.median(r[a] + r[f'err_{a}'] for r in leg) for a in 'xyz']
    start = [leg[0][a] for a in 'xyz']
    d = [g - s for g, s in zip(tgt, start)]
    dist = math.sqrt(sum(v * v for v in d)) or 1e-9
    u = [v / dist for v in d]
    err = [math.dist([r[a] for a in 'xyz'], tgt) for r in leg]
    arrive = {}
    for thr in (0.10, 0.05):
        hit = next((i for i, e in enumerate(err) if e < thr), None)
        arrive[thr] = (leg[hit]['drone_ms'] - t0) / 1000 if hit is not None else None
    past = max(sum((r[a] - g) * ui for a, g, ui in zip('xyz', tgt, u)) for r in leg)
    t_end = leg[-1]['drone_ms']
    tail = [e for r, e in zip(leg, err) if r['drone_ms'] >= t_end - 500]
    # speed over 5-sample spans (0.1 s), which averages out estimate jitter
    smooth = [math.dist([b[k] for k in 'xyz'], [a[k] for k in 'xyz']) / ((b['drone_ms'] - a['drone_ms']) / 1000)
              for a, b in zip(leg, leg[5:]) if b['drone_ms'] > a['drone_ms']] or [0.0]
    return {'jump_m': dist, 'arrive10_s': arrive[0.10], 'arrive5_s': arrive[0.05],
            'overshoot_cm': 100 * max(0.0, past), 'final_cm': 100 * st.mean(tail), 'peak_speed': max(smooth),
            'leg_s': (t_end - t0) / 1000}


def report(path: Path, label: str) -> tuple[dict, list[dict]]:
    rows = rows_of(path)
    active = [r for r in rows if r['policy_active'] == 1 and r['err_x'] is not None]
    reached = {r['phase'] for r in rows}
    out = {'label': label, 'log': path.name, 'policy_s': round(len(active) * 0.02, 1),
           'completed': 'landed' in reached or 'ground' in reached}
    for ph in PHASES:
        e = [100 * math.sqrt(r['err_x'] ** 2 + r['err_y'] ** 2 + r['err_z'] ** 2) for r in active if r['phase'] == ph]
        out[f'{ph}_cm'] = round(st.mean(e), 1) if e else None
        out[f'{ph}_max_cm'] = round(max(e), 1) if e else None
    # Smoothness over the airborne phases (average error misses rocking around the right path,
    # 2026-11-25): tilt and rotation rate RMS, wobble (RMS distance from the 0.5 s moving average
    # of the position), acceleration RMS (second difference over 0.1 s).
    air = [r for r in active if r['phase'] in ('hover', 'move', 'circle', 'waypoint', 'back', 'hover2')]
    tilt = [math.hypot(r['roll_deg'], r['pitch_deg']) for r in air if r.get('roll_deg') is not None]
    gyro = [math.sqrt(sum(r[k] ** 2 for k in ('gyro_x', 'gyro_y', 'gyro_z'))) for r in air
            if r.get('gyro_x') is not None]
    pos = [(r['x'], r['y'], r['z']) for r in air]
    if tilt:
        out['tilt_rms_deg'] = round(math.sqrt(st.mean(v * v for v in tilt)), 1)
        out['tilt_max_deg'] = round(max(tilt), 1)
    if gyro:
        out['gyro_rms_dps'] = round(math.sqrt(st.mean(v * v for v in gyro)))
    if len(pos) > 30:
        wob = [math.dist(pos[i], [st.mean(p[k] for p in pos[i - 12:i + 13]) for k in range(3)])
               for i in range(12, len(pos) - 12)]
        out['wobble_cm'] = round(100 * math.sqrt(st.mean(v * v for v in wob)), 2)
        acc = [math.dist([pos[i + 5][k] - 2 * pos[i][k] + pos[i - 5][k] for k in range(3)], [0, 0, 0]) / 0.01
               for i in range(5, len(pos) - 5)]
        out['accel_rms'] = round(math.sqrt(st.mean(a * a for a in acc)), 2)
    hov = [r for r in active if r['phase'] in ('hover', 'hover2')]
    if len(hov) > 5:
        out['jitter'] = round(st.mean(st.pstdev(r[f'm{m}'] / 65535 for r in hov) for m in range(1, 5)), 4)
    first = next((r for r in rows if r['x'] is not None), None)
    last = [r for r in rows if r['phase'] in ('landed', 'ground') and r['x'] is not None]
    if first and last:
        out['landing_cm'] = round(100 * math.hypot(last[-1]['x'] - first['x'], last[-1]['y'] - first['y']), 1)
    # The target is rebuilt as estimate + error. The two come from different log blocks, a few
    # ms apart, so in fast flight the rebuilt target jitters by several cm: smooth it with a
    # running median (9 samples, 0.18 s), then start a new leg only when it moves more than 8 cm
    # (the smallest jump flown is 10 cm).
    # 'back' is a waypoint leg only in a --waypoints mission; in a circle mission it is a ramp
    wp = [r for r in active if r['phase'] in ('waypoint', 'back')] if 'waypoint' in reached else []
    raw = [[r[a] + r[f'err_{a}'] for a in 'xyz'] for r in wp]
    smooth = [[st.median(p[i] for p in raw[max(0, j - 4):j + 5]) for i in range(3)] for j in range(len(raw))]
    legs, cur, key = [], [], None
    for r, k in zip(wp, smooth):
        if key is not None and math.dist(k, key) > 0.08 and cur:
            legs.append(cur)
            cur = []
        if not cur:
            key = k
        cur.append(r)
    if cur:
        legs.append(cur)
    leg_rows = [{'label': label, 'log': path.name, 'leg': i, **leg_metrics(leg)} for i, leg in enumerate(legs) if len(leg) > 10]
    if leg_rows:
        a10 = [lr['arrive10_s'] for lr in leg_rows if lr['arrive10_s'] is not None]
        out['wp_legs'] = len(leg_rows)
        out['wp_arrived10_pct'] = round(100 * len(a10) / len(leg_rows))
        out['wp_arrive10_s'] = round(st.median(a10), 2) if a10 else None
        out['wp_overshoot_cm'] = round(st.median(lr['overshoot_cm'] for lr in leg_rows), 1)
        out['wp_final_cm'] = round(st.median(lr['final_cm'] for lr in leg_rows), 1)
        out['wp_peak_speed'] = round(max(lr['peak_speed'] for lr in leg_rows), 2)
    return out, leg_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('logs', nargs='+', type=Path, help='rl_trigger flight CSVs')
    ap.add_argument('--label', default='', help='policy name for the table (e.g. svdpga)')
    a = ap.parse_args()
    flights, legs = [], []
    for p in a.logs:
        f, lr = report(p, a.label)
        flights.append(f)
        legs += lr
        parts = [f'{k} {v}' for k, v in f.items() if k not in ('label', 'log') and v is not None and not k.endswith('_max_cm')]
        log.info('%s [%s]: %s', p.name, a.label, ', '.join(parts))
        for r in lr:
            log.info('  leg %d: jump %.2f m, 10 cm in %s s, 5 cm in %s s, overshoot %.1f cm, final %.1f cm, peak %.2f m/s',
                     r['leg'], r['jump_m'], r['arrive10_s'], r['arrive5_s'], r['overshoot_cm'], r['final_cm'], r['peak_speed'])
    keys = list(dict.fromkeys(k for f in flights for k in f))
    out = run_log.csv_path('flights', script=__file__, when=STARTED_AT)
    run_log.write_csv(out, keys, [[f.get(k) for k in keys] for f in flights], script=__file__, generated_at=STARTED_AT)
    if legs:
        lout = run_log.csv_path('legs', script=__file__, when=STARTED_AT)
        lkeys = list(legs[0])
        run_log.write_csv(lout, lkeys, [[r[k] for k in lkeys] for r in legs], script=__file__, generated_at=STARTED_AT)
    log.info('wrote %s', out)


if __name__ == '__main__':
    main()
