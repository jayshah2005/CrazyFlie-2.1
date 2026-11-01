"""
Fit a brushless motor model from a sysid_hover.py flight log.

Per motor, with u = command / 65535 and samples every dt:

    rpm[k+1] = rpm[k] + dt / tau * (K * u[k] + b - rpm[k])

i.e. a first-order lag (time constant tau) towards a linear steady-state map
rpm_ss = K * u + b. Also reports the command-to-RPM delay, hover RPM and the
thrust coefficient implied by the drone mass.

Usage: python sysid_analyze.py [log.csv] [--mass-g 41.3]
"""
import argparse
import csv
import glob
import math
import os

import numpy as np

import run_log

log = run_log.setup(__file__)

AIRBORNE = ('hover', 'step_up', 'step_down')
IDSIA_KF = 3.72e-8  # N/(rad/s)^2, Busetto et al. (arXiv 2512.14450)


INVALID_RPM = 40000  # eRPM telemetry reports failed reads as 65535


def load(path):
    """Read the log; replace invalid RPM reads by the previous valid value."""
    with open(path) as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith('#'))
                if r['vbat']]
    for i in range(1, 5):
        key, last, bad = f'rpm{i}', 0.0, 0
        for r in rows:
            v = float(r[key])
            if v >= INVALID_RPM:
                r[key], bad = last, bad + 1
            else:
                last = v
        log.info('Motor %d: replaced %d invalid RPM reads', i, bad)
    return rows


def lowpass(x, alpha):
    y = np.empty_like(x)
    y[0] = x[0]
    for k in range(1, len(x)):
        y[k] = y[k - 1] + alpha * (x[k - 1] - y[k - 1])
    return y


def fit_motor(t, u, rpm, max_delay=12, taus=np.arange(0.005, 0.2005, 0.005)):
    """Output-error fit of: rpm = K * lowpass_tau(u delayed by d samples) + b.

    The model is simulated from the commands and compared with the whole
    measured RPM trace, which is robust to RPM measurement noise. K and b are
    linear given (d, tau); (d, tau) are grid-searched.
    """
    dt = float(np.median(np.diff(t)))
    best = None
    for d in range(max_delay + 1):
        u_d = np.concatenate([np.full(d, u[0]), u[:len(u) - d]])
        for tau in taus:
            x = lowpass(u_d, min(1.0, dt / tau))
            A = np.column_stack([x, np.ones(len(x))])
            (K, b), *_ = np.linalg.lstsq(A, rpm, rcond=None)
            resid = rpm - A @ np.array([K, b])
            r2 = 1 - np.sum(resid ** 2) / np.sum((rpm - rpm.mean()) ** 2)
            if best is None or r2 > best['r2']:
                best = {'tau': float(tau), 'K': K, 'b': b, 'r2': r2, 'dt': dt,
                        'delay': d * dt}
    return best


def delay(t, u, rpm, max_lag_s=0.1):
    """Lag (s) that best aligns the command changes with the RPM changes."""
    dt = float(np.median(np.diff(t)))
    du, dr = np.diff(u), np.diff(rpm)
    du, dr = du - du.mean(), dr - dr.mean()
    lags = range(0, int(max_lag_s / dt) + 1)
    score = [np.dot(du[:len(du) - k], dr[k:]) for k in lags]
    return dt * int(np.argmax(score))


def main(args):
    path = args.log or max(glob.glob('logs/sysid_hover_sysid_*.csv'), key=os.path.getmtime)
    rows = load(path)
    log.info('Log: %s (%d samples)', path, len(rows))
    air = [r for r in rows if r['phase'].startswith(AIRBORNE)]
    t = np.array([int(r['drone_ms']) for r in air]) / 1000.0
    vbat = np.array([float(r['vbat']) for r in air])
    log.info('Battery while airborne: %.2f V mean, %.2f V min', vbat.mean(), vbat.min())

    fits = []
    for i in range(1, 5):
        u = np.array([float(r[f'm{i}']) for r in air]) / 65535.0
        rpm = np.array([float(r[f'rpm{i}']) for r in air])
        f = fit_motor(t, u, rpm)
        f['xcorr_delay'] = delay(t, u, rpm)
        f['u_range'] = (u.min(), u.max())
        fits.append(f)
        log.info('Motor %d: delay=%.0f ms then tau=%.3f s, rpm_ss = %.0f*u %+.0f (R2=%.2f); '
                 'xcorr delay %.0f ms; u %.2f..%.2f', i, 1000 * f['delay'], f['tau'], f['K'],
                 f['b'], f['r2'], 1000 * f['xcorr_delay'], *f['u_range'])
    tau = float(np.median([f['tau'] for f in fits]))
    delay_s = float(np.median([f['delay'] for f in fits]))
    K = float(np.mean([f['K'] for f in fits]))
    b = float(np.mean([f['b'] for f in fits]))
    log.info('Mean model: delay=%.0f ms, tau=%.3f s, rpm_ss = %.0f*u %+.0f, full command -> '
             '%.0f RPM (at %.2f V), dt=%.3f s', 1000 * delay_s, tau, K, b, K + b, vbat.mean(),
             fits[0]['dt'])
    # A first-order lag with time constant T_eq ~ delay + tau reproduces the
    # same mean lag; the paper's simulator only has the lag term.
    log.info('Equivalent first-order time constant (delay + tau): %.3f s '
             '(brushed model used 0.150 s)', delay_s + tau)
    if tau < 2 * fits[0]['dt']:
        log.warning('tau is under 2 samples: the motors respond faster than the 100 Hz '
                    'log resolves; treat tau as an upper bound.')

    hover = [r for r in rows if r['phase'] == 'hover']
    rpm_hover = np.mean([[float(r[f'rpm{i}']) for i in range(1, 5)] for r in hover])
    u_hover = np.mean([[float(r[f'm{i}']) for i in range(1, 5)] for r in hover]) / 65535
    omega = rpm_hover * 2 * math.pi / 60
    kf = args.mass_g / 1000 * 9.81 / (4 * omega ** 2)
    log.info('Hover: %.0f RPM at command %.1f%%; implied thrust coefficient %.2e N/(rad/s)^2 '
             '(IDSIA: %.2e, ratio %.2f)', rpm_hover, 100 * u_hover, kf, IDSIA_KF, kf / IDSIA_KF)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('log', nargs='?', help='flight CSV (default: newest sysid log)')
    parser.add_argument('--mass-g', type=float, default=41.3, help='take-off mass in grams')
    main(parser.parse_args())
