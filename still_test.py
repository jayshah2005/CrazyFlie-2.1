"""
Still test: leave the drone untouched on the floor and measure how much the
state estimate moves. A still drone should give near-constant numbers, so the
spread is the estimator noise (Lighthouse + IMU).

Prints mean / std / peak-to-peak per variable and the Lighthouse status, and
writes the raw samples to a CSV under logs/.

Usage: python still_test.py [--seconds 10]
"""
import argparse
import statistics
import time
from datetime import datetime

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

import run_log

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7EA')
log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()
LOG_FILE = run_log.csv_path('still', script=__file__, when=STARTED_AT)

POSE = ['stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
        'stateEstimate.roll', 'stateEstimate.pitch', 'stateEstimate.yaw']
EXTRA = [('gyro.x', 'float'), ('gyro.y', 'float'), ('gyro.z', 'float'),
         ('pm.vbat', 'float'), ('lighthouse.status', 'uint8_t'),
         ('lighthouse.bsReceive', 'uint16_t')]
# (unit, what counts as "quiet" for a drone sitting still)
LIMITS = {'x': ('m', 0.005), 'y': ('m', 0.005), 'z': ('m', 0.005),
          'roll': ('deg', 0.3), 'pitch': ('deg', 0.3), 'yaw': ('deg', 0.5),
          'gyro.x': ('deg/s', 0.5), 'gyro.y': ('deg/s', 0.5), 'gyro.z': ('deg/s', 0.5)}
LH_STATUS = {0: 'not receiving', 1: 'receiving, not estimating', 2: 'estimating position'}


def main(seconds):
    cflib.crtp.init_drivers()
    log.info('Connecting to %s ...', URI)
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        toc = scf.cf.log.toc
        extra = [(n, t) for n, t in EXTRA
                 if toc.get_element_by_complete_name(n) is not None]
        rows, latest = [], {}

        a = LogConfig(name='StillPose', period_in_ms=20)
        for name in POSE:
            a.add_variable(name, 'float')
        b = LogConfig(name='StillExtra', period_in_ms=20)
        for name, kind in extra:
            b.add_variable(name, kind)
        a.data_received_cb.add_callback(
            lambda t, d, c: rows.append([t] + [d[n] for n in POSE]
                                        + [latest.get(n) for n, _ in extra]))
        b.data_received_cb.add_callback(lambda t, d, c: latest.update(d))
        for conf in (a, b):
            scf.cf.log.add_config(conf)
            conf.start()

        log.info('Recording %.0f s. Do not touch the drone.', seconds)
        time.sleep(seconds)
        for conf in (a, b):
            conf.stop()

    fields = ['drone_ms'] + POSE + [n for n, _ in extra]
    run_log.write_csv(LOG_FILE, fields, rows, script=__file__,
                      generated_at=datetime.now().astimezone())
    log.info('%s samples written to %s', len(rows), LOG_FILE)
    report(fields, rows)


def report(fields, rows):
    rows = [r for r in rows if None not in r]
    if len(rows) < 10:
        log.error('Too few samples to analyse.')
        return
    col = {f: [r[i] for r in rows] for i, f in enumerate(fields)}
    log.info('%-8s %10s %10s %12s   verdict', 'value', 'mean', 'std', 'peak-peak')
    for full in POSE + ['gyro.x', 'gyro.y', 'gyro.z']:
        if full not in col:
            continue
        key = full.split('.', 1)[1] if full.startswith('stateEstimate') else full
        unit, quiet = LIMITS[key]
        v = col[full]
        std = statistics.pstdev(v)
        log.info('%-8s %10.3f %10.4f %12.4f   %s %s', key, statistics.fmean(v),
                 std, max(v) - min(v), unit, 'ok' if std <= quiet else 'NOISY')
    if 'pm.vbat' in col:
        log.info('battery  %.2f V', statistics.fmean(col['pm.vbat']))
    if 'lighthouse.status' in col:
        counts = {s: col['lighthouse.status'].count(s) for s in set(col['lighthouse.status'])}
        for s, n in sorted(counts.items()):
            log.info('lighthouse status %s (%s): %.0f%% of samples',
                     s, LH_STATUS.get(s, '?'), 100 * n / len(rows))
    if 'lighthouse.bsReceive' in col:
        seen = 0
        for mask in col['lighthouse.bsReceive']:
            seen |= int(mask)
        log.info('base stations received: %s',
                 [i for i in range(16) if seen & (1 << i)] or 'none')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--seconds', type=float, default=10.0)
    main(parser.parse_args().seconds)
