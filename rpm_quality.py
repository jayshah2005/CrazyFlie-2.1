"""
Check how good the measured motor speeds (motor.m*_rpm) are as a policy input.

Reads existing flight logs (no flying) and reports, per motor, while the
motors are spinning:

  - units: mechanical RPM or electrical RPM (eRPM = RPM x pole pairs), from
    the hover RPM the drone's weight requires;
  - invalid reads (65535) and zero reads: rate and longest burst;
  - stale reads: RPM identical to the previous sample while the command moved;
  - noise: sample-to-sample jitter around a short moving average;
  - lag: command -> measured RPM, fitted as delay + first-order lag on
    u * Vbat (u = command / 65535), with the residual the simulator would
    have to cover as sensor noise.

sysid_hover logs (10 ms, command and RPM in the same log packet) give the
cleanest timing; rl_trigger logs (20 ms) show the sensor during policy flights.

Usage: python rpm_quality.py [log.csv ...] [--mass-g 41.3]
       (default: every sysid_hover and rl_trigger log in logs/)
"""
import argparse
import csv
import glob
import math
from datetime import datetime

import numpy as np

import run_log
from sysid_analyze import fit_motor, lowpass

log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()

INVALID_RPM = 40000      # failed eRPM reads come through as 65535
RUNNING_CMD = 10000      # all four commands above this = motors spinning
MIN_SEGMENT_S = 1.0      # ignore shorter spinning stretches
SIM_RPM_HALF_RANGE = 12000  # simulator: rotor input = rpm / 12000 - 1 (limit 0..24000)
IDSIA_KF = 3.72e-8       # N/(rad/s)^2, Busetto et al. (arXiv 2512.14450), independent of our logs
SUMMARY_FIELDS = ['log', 'kind', 'motor', 'dt_ms', 'samples', 'invalid_pct', 'invalid_max_run',
                  'zero_pct', 'stale_pct', 'stale_max_run', 'jitter_rpm', 'delay_ms', 'tau_s',
                  'r2', 'model_resid_rpm', 'mean_rpm']


def load(path):
    """Rows of a log, with the command column used for each motor."""
    with open(path) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith('#')))
    kind = 'sysid' if 'sysid_hover' in path else 'rl_trigger'
    # rl_trigger: out* = ratio sent to the ESCs (m* is the policy's own output)
    cmd = 'm' if kind == 'sysid' else 'out'
    return kind, cmd, rows


def segments(rows, cmd):
    """Stretches of consecutive rows where all four motors are spinning."""
    segs, cur = [], []
    for r in rows:
        try:
            ok = (r['vbat'] and r['rpm1'] != ''
                  and all(float(r[f'{cmd}{i}']) >= RUNNING_CMD for i in range(1, 5)))
        except (TypeError, ValueError):
            ok = False
        if ok:
            cur.append(r)
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    out = []
    for s in segs:
        t = np.array([int(r['drone_ms']) for r in s]) / 1000.0
        if t[-1] - t[0] >= MIN_SEGMENT_S:
            out.append(s)
    return out


def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def analyse_motor(seg_rows, cmd, i):
    """Per-motor numbers pooled over all spinning segments of one log."""
    raw_all, fills, fits, jit, resid = [], [], [], [], []
    stale, stale_runs, dts = 0, [], []
    for s in seg_rows:
        t = np.array([int(r['drone_ms']) for r in s]) / 1000.0
        u = np.array([float(r[f'{cmd}{i}']) for r in s]) / 65535.0
        vbat = np.array([float(r['vbat']) for r in s])
        raw = np.array([float(r[f'rpm{i}']) for r in s])
        raw_all.append(raw)
        dts.append(np.median(np.diff(t)))

        bad = (raw >= INVALID_RPM) | (raw == 0)
        rpm = raw.copy()
        for k in range(1, len(rpm)):  # hold the last good value, as firmware would
            if bad[k]:
                rpm[k] = rpm[k - 1]
        fills.append(rpm)

        # Stale: a good read equal to the previous good read while the command moved.
        # A fully repeated row (command and RPM unchanged) is a logging artefact, not counted.
        same = np.zeros(len(raw), bool)
        same[1:] = (~bad[1:]) & (~bad[:-1]) & (raw[1:] == raw[:-1]) & (u[1:] != u[:-1])
        stale += int(same.sum())
        stale_runs.append(longest_run(same))

        # Jitter: deviation from a centred 5-sample moving average.
        if len(rpm) >= 5:
            ma = np.convolve(rpm, np.ones(5) / 5, mode='valid')
            jit.append(rpm[2:-2] - ma)

        f = fit_motor(t, u * vbat, rpm)
        resid.append(rpm - fitted_trace(t, u * vbat, f))
        fits.append((len(rpm), f))

    raw = np.concatenate(raw_all)
    n = len(raw)
    w = np.array([k for k, _ in fits], float)
    return {
        'dt_ms': 1000 * float(np.median(dts)),
        'samples': n,
        'invalid_pct': 100 * float(np.mean(raw >= INVALID_RPM)),
        'invalid_max_run': max(longest_run(r >= INVALID_RPM) for r in raw_all),
        'zero_pct': 100 * float(np.mean(raw == 0)),
        'stale_pct': 100 * stale / n,
        'stale_max_run': max(stale_runs),
        'jitter_rpm': float(np.std(np.concatenate(jit))) if jit else float('nan'),
        'delay_ms': 1000 * float(np.average([f['delay'] for _, f in fits], weights=w)),
        'tau_s': float(np.average([f['tau'] for _, f in fits], weights=w)),
        'r2': float(np.average([f['r2'] for _, f in fits], weights=w)),
        'model_resid_rpm': float(np.std(np.concatenate(resid))),
        'mean_rpm': float(np.mean(np.concatenate(fills))),
    }


def fitted_trace(t, x, f):
    """Model RPM for a fit from sysid_analyze.fit_motor (delay + lag + linear map)."""
    d = int(round(f['delay'] / f['dt']))
    x_d = np.concatenate([np.full(d, x[0]), x[:len(x) - d]])
    return f['K'] * lowpass(x_d, min(1.0, f['dt'] / f['tau'])) + f['b']


def main(args):
    paths = args.logs or sorted(glob.glob('logs/sysid_hover_sysid_*.csv')
                                + glob.glob('logs/rl_trigger_flight_*.csv'))
    table, hover_means = [], []
    for path in paths:
        kind, cmd, rows = load(path)
        if not rows or 'rpm1' not in rows[0]:
            log.info('%s: no RPM columns (older log), skipped', path)
            continue
        segs = segments(rows, cmd)
        if not segs:
            log.info('%s: no stretch of >= %.0f s with all motors spinning, skipped',
                     path, MIN_SEGMENT_S)
            continue
        dur = sum((int(s[-1]['drone_ms']) - int(s[0]['drone_ms'])) / 1000 for s in segs)
        log.info('%s (%s): %.1f s spinning in %d stretch(es)', path, kind, dur, len(segs))
        for i in range(1, 5):
            m = analyse_motor(segs, cmd, i)
            table.append({'log': path, 'kind': kind, 'motor': i, **m})
            log.info('  motor %d: invalid %.2f%% (run %d), zero %.2f%%, stale %.1f%% (run %d), '
                     'jitter %.0f RPM, delay %.0f ms + tau %.3f s, R2 %.2f, model resid %.0f RPM',
                     i, m['invalid_pct'], m['invalid_max_run'], m['zero_pct'], m['stale_pct'],
                     m['stale_max_run'], m['jitter_rpm'], m['delay_ms'], m['tau_s'], m['r2'],
                     m['model_resid_rpm'])
        if kind == 'sysid':
            hover = [r for r in rows if r['phase'] == 'hover']
            vals = [float(r[f'rpm{i}']) for r in hover for i in range(1, 5)
                    if r[f'rpm{i}'] and 0 < float(r[f'rpm{i}']) < INVALID_RPM]
            if vals:
                hover_means.append(float(np.mean(vals)))

    if not table:
        log.error('No usable logs.')
        return

    path = run_log.csv_path('summary', script=__file__, when=STARTED_AT)
    written = run_log.write_csv(path, SUMMARY_FIELDS, [[r[k] for k in SUMMARY_FIELDS]
                                                       for r in table],
                                script=__file__, generated_at=datetime.now().astimezone())
    log.info('Per-log summary written to %s', written)

    # ---------- overall verdict ----------
    def pooled(kind, key, agg=np.median):
        v = [r[key] for r in table if r['kind'] == kind]
        return float(agg(v)) if v else float('nan')

    log.info('--- Overall (median over motors and logs) ---')
    for kind in ('sysid', 'rl_trigger'):
        if not any(r['kind'] == kind for r in table):
            continue
        log.info('%s: invalid %.2f%% (worst %.2f%%, longest run %d), zero %.2f%%, '
                 'stale %.1f%%, jitter %.0f RPM, delay %.0f ms, tau %.3f s, model resid %.0f RPM',
                 kind, pooled(kind, 'invalid_pct'), pooled(kind, 'invalid_pct', np.max),
                 int(pooled(kind, 'invalid_max_run', np.max)), pooled(kind, 'zero_pct'),
                 pooled(kind, 'stale_pct'), pooled(kind, 'jitter_rpm'),
                 pooled(kind, 'delay_ms'), pooled(kind, 'tau_s'),
                 pooled(kind, 'model_resid_rpm'))

    if hover_means:
        rpm_hover = float(np.mean(hover_means))
        omega = math.sqrt(args.mass_g / 1000 * 9.81 / 4 / IDSIA_KF)
        expected = omega * 60 / (2 * math.pi)
        ratio = rpm_hover / expected
        units = 'mechanical RPM' if 0.8 < ratio < 1.25 else (
            f'electrical RPM (about {ratio:.1f} x mechanical)')
        log.info('Units: hover reads %.0f, weight needs %.0f mechanical RPM (IDSIA thrust '
                 'coefficient) -> ratio %.2f -> %s', rpm_hover, expected, ratio, units)

    j = pooled('sysid', 'jitter_rpm')
    if math.isnan(j):
        j = pooled('rl_trigger', 'jitter_rpm')
    log.info('In simulator units (rpm / %d - 1): jitter std %.4f, model residual std %.4f',
             SIM_RPM_HALF_RANGE, j / SIM_RPM_HALF_RANGE,
             pooled('rl_trigger', 'model_resid_rpm') / SIM_RPM_HALF_RANGE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('logs', nargs='*', help='flight CSVs (default: all sysid + rl_trigger)')
    parser.add_argument('--mass-g', type=float, default=41.3, help='take-off mass in grams')
    main(parser.parse_args())
