"""
Lighthouse diagnostic: is the drone seeing the base stations?

Leave the drone still where the base stations can see it (top of the deck
facing them). Close cfclient first: only one program can use the radio.

Reports:
  - whether the Lighthouse deck is detected
  - per base station: how often its sweeps are received, whether it is
    "active" (used by the estimator), and whether calibration and geometry
    are stored on the drone
  - the overall Lighthouse status (not receiving / receiving / estimating)
and writes the raw samples to a CSV under logs/.

Usage: python lighthouse_check.py [--seconds 5]
"""
import argparse
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
LOG_FILE = run_log.csv_path('lighthouse', script=__file__, when=STARTED_AT)

# Bit masks: bit i = base station i (channel i + 1 on a V2 base station).
LH_VARS = [('lighthouse.status', 'uint8_t'), ('lighthouse.bsReceive', 'uint16_t'),
           ('lighthouse.bsActive', 'uint16_t'), ('lighthouse.bsCalVal', 'uint16_t'),
           ('lighthouse.bsGeoVal', 'uint16_t'), ('pm.vbat', 'float')]
POSE = [('stateEstimate.x', 'float'), ('stateEstimate.y', 'float'),
        ('stateEstimate.z', 'float')]
LH_STATUS = {0: 'not receiving', 1: 'receiving, not estimating', 2: 'estimating position'}
MAX_BS = 16


def param(cf, name):
    try:
        return cf.param.get_value(name)
    except (KeyError, AttributeError):
        return None


def main(seconds):
    cflib.crtp.init_drivers()
    log.info('Connecting to %s ... (cfclient must be closed)', URI)
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        deck = param(cf, 'deck.bcLighthouse4')
        if deck is None:
            log.error('Firmware has no Lighthouse deck driver parameter (deck.bcLighthouse4).')
        elif int(deck) == 0:
            log.error('Lighthouse deck NOT detected. Reseat the deck (pins fully in, '
                      'correct orientation) and power-cycle.')
        else:
            log.info('Lighthouse deck detected.')

        toc = cf.log.toc
        names = [(n, t) for n, t in LH_VARS + POSE
                 if toc.get_element_by_complete_name(n) is not None]
        missing = [n for n, _ in LH_VARS if toc.get_element_by_complete_name(n) is None]
        if missing:
            log.warning('Not in this firmware\'s log TOC: %s', ', '.join(missing))

        rows = []
        conf = LogConfig(name='LhCheck', period_in_ms=50)
        for n, t in names:
            conf.add_variable(n, t)
        conf.data_received_cb.add_callback(
            lambda ts, d, c: rows.append([ts] + [d[n] for n, _ in names]))
        cf.log.add_config(conf)
        conf.start()
        log.info('Recording %.0f s. Keep the drone still and in view of the base stations.',
                 seconds)
        time.sleep(seconds)
        conf.stop()

    fields = ['drone_ms'] + [n for n, _ in names]
    run_log.write_csv(LOG_FILE, fields, rows, script=__file__,
                      generated_at=datetime.now().astimezone())
    log.info('%s samples written to %s', len(rows), LOG_FILE)
    report(fields, rows)


def report(fields, rows):
    if not rows:
        log.error('No samples received from the drone.')
        return
    col = {f: [r[i] for r in rows] for i, f in enumerate(fields)}
    n = len(rows)

    def ever(name):
        m = 0
        for v in col.get(name, []):
            m |= int(v)
        return m

    def share(name, bs):
        return 100 * sum(1 for v in col.get(name, []) if int(v) & (1 << bs)) / n

    rx, act = ever('lighthouse.bsReceive'), ever('lighthouse.bsActive')
    cal, geo = ever('lighthouse.bsCalVal'), ever('lighthouse.bsGeoVal')
    stations = [i for i in range(MAX_BS) if (rx | act | cal | geo) & (1 << i)]
    if not stations:
        log.error('No base station seen, and none has stored calibration or geometry.')
    log.info('%-4s %-8s %10s %10s %12s %10s', 'bs', 'channel', 'received', 'active',
             'calibration', 'geometry')
    for i in stations:
        log.info('%-4d %-8d %9.0f%% %9.0f%% %12s %10s', i, i + 1,
                 share('lighthouse.bsReceive', i), share('lighthouse.bsActive', i),
                 'ok' if cal & (1 << i) else 'MISSING', 'ok' if geo & (1 << i) else 'MISSING')

    if 'lighthouse.status' in col:
        counts = {s: col['lighthouse.status'].count(s) for s in set(col['lighthouse.status'])}
        for s, k in sorted(counts.items()):
            log.info('lighthouse status %s (%s): %.0f%% of samples', s, LH_STATUS.get(s, '?'),
                     100 * k / n)
    if 'stateEstimate.x' in col:
        log.info('position estimate (last): x %.3f  y %.3f  z %.3f', col['stateEstimate.x'][-1],
                 col['stateEstimate.y'][-1], col['stateEstimate.z'][-1])
    if 'pm.vbat' in col:
        log.info('battery %.2f V', col['pm.vbat'][-1])

    # Plain-language diagnosis.
    if not rx:
        log.warning('DIAGNOSIS: the deck receives no sweeps at all. Check: base stations '
                    'powered (green LED, not blinking), drone in their field of view (about '
                    '120 deg cone, no obstruction above the deck), not in direct sunlight, '
                    'and the deck LEDs.')
    elif rx and not (cal & rx):
        log.warning('DIAGNOSIS: sweeps received but calibration missing. Leave the drone in '
                    'view ~10 s (calibration is broadcast by the base station), then re-run.')
    elif rx and not (geo & rx):
        log.warning('DIAGNOSIS: sweeps received but no geometry stored for those stations. '
                    'Run the geometry estimation in cfclient (Lighthouse tab -> Manage '
                    'geometry) with the drone at the origin, and write it to the drone.')
    elif rx and not act:
        log.warning('DIAGNOSIS: received with calibration and geometry, but no station is used '
                    'by the estimator. Check stabilizer.estimator = 2 (Kalman) and '
                    'power-cycle.')
    elif len([i for i in stations if share('lighthouse.bsReceive', i) > 80]) < 2:
        log.warning('DIAGNOSIS: fewer than 2 stations received reliably. Position works with '
                    'one, but is much better with two; check the other one\'s view / channel.')
    else:
        log.info('DIAGNOSIS: Lighthouse looks healthy on the drone side. If cfclient still '
                 'shows nothing, the problem is in cfclient (try reconnecting, restarting it).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--seconds', type=float, default=5.0)
    main(parser.parse_args().seconds)
