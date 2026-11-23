"""
Sensor check, no flying: is a problem in the Lighthouse deck, a base station, the stored
geometry, or another sensor?

Records, with the motors off:
  - the sweep angles of all 4 photodiodes of the Lighthouse deck from every base station
    (the locSrv angle stream that cfclient's geometry estimation uses), calibrated, V1 form
  - the position estimate, Lighthouse status, gyro, accelerometer, barometer, range sensor,
    optical flow and battery

Steps (Enter between them; the drone at the take-off spot unless said otherwise):
  floor_0, floor_90, floor_180, floor_270   still on the floor, turned in place 90 deg
                                            counter-clockwise (seen from above) each time
  raised                                    still on a stable support 0.5-1 m high, facing +x
                                            (chair, box; skip with --no-raised)
  lift                                      held by hand: lift slowly to ~1 m, hold, lower

How the results separate the causes:
  base station: one station's row is bad (rate, noise, outliers) on all 4 photodiodes, in
                every orientation
  deck:         one photodiode's column is bad for both stations, and follows the
                photodiode when the drone turns
  geometry:     noise low, but the rays of the two stations to the same photodiode miss
                each other by more than ~1 cm, or the photodiode positions they give do
                not form the deck's 30 x 15 mm rectangle (the stored geometry or
                calibration is wrong)
  occlusion / reflection: rate or outliers depend on where the drone is and which way
                it faces, not on one station or one photodiode
Other sensors: noise at rest against today's survey (l2f/sim/noise_measured.json), and in
the lift the range sensor against the Lighthouse height (two independent sensors).

Writes logs/sensor_check_angles_<time>.csv, logs/sensor_check_telemetry_<time>.csv and the
drone's geometry to logs/sensor_check_geometry_<time>.json.

Usage: python sensor_check.py [--seconds 15] [--no-raised] [--no-lift]
       python sensor_check.py --steps raised_0 raised_180   (blocked line of sight: room or drone?)
       python sensor_check.py --analyze logs/sensor_check_angles_<time>.csv
"""
import argparse
import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import run_log

log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()
ROOT = Path(__file__).resolve().parent
NOISE_FILE = ROOT / 'l2f' / 'sim' / 'noise_measured.json'
DEFAULT_URI = 'radio://0/80/2M/E7E7E7E7EA'
N_SENSORS = 4
# Lighthouse deck photodiodes in the drone frame (cflib LhDeck4SensorPositions), m
SENSOR_POS = np.array([(-0.015, 0.0075, 0.0), (-0.015, -0.0075, 0.0), (0.015, 0.0075, 0.0), (0.015, -0.0075, 0.0)])
PAIR_MS = 15          # pair the two stations' packets at most this far apart (one rotation is 20 ms)
STALE_RAD = 2e-6      # a photodiode angle that moved less than this since the last packet is stale
                      # (the firmware resends the old angle when a photodiode missed the sweep;
                      # photodiodes 1-3 come as fp16 differences, about 4-8e-6 rad resolution, so
                      # their stale share is approximate)
OUTLIER_MAD = 8       # an angle more than this many MADs from the segment median is an outlier
TELEMETRY = [  # (log variable, type, column); blocks of at most 26 bytes
    [('stateEstimate.x', 'float', 'x'), ('stateEstimate.y', 'float', 'y'), ('stateEstimate.z', 'float', 'z'),
     ('stateEstimate.yaw', 'float', 'yaw_deg'), ('lighthouse.bsReceive', 'uint16_t', 'lh_receive'),
     ('lighthouse.bsActive', 'uint16_t', 'lh_active'), ('lighthouse.status', 'uint8_t', 'lh_status')],
    [('gyro.x', 'float', 'gyro_x'), ('gyro.y', 'float', 'gyro_y'), ('gyro.z', 'float', 'gyro_z'),
     ('acc.x', 'float', 'acc_x'), ('acc.y', 'float', 'acc_y'), ('acc.z', 'float', 'acc_z')],
    [('range.zrange', 'uint16_t', 'zrange_mm'), ('motion.deltaX', 'int16_t', 'flow_dx'),
     ('motion.deltaY', 'int16_t', 'flow_dy'), ('motion.squal', 'uint8_t', 'flow_squal'),
     ('baro.pressure', 'float', 'baro_hpa'), ('pm.vbat', 'float', 'vbat')],
]
TEL_COLUMNS = ['pc_s', 'step', 'block'] + [c for b in TELEMETRY for _, _, c in b]
AWAY = ' Nobody between the drone and either base station (stand behind one).'
STEPS = {
    'floor_0': 'Put the drone at the take-off spot facing +x, motors off. Do not touch it.',
    'floor_90': 'Turn it 90 deg counter-clockwise (seen from above) in place: it faces +y.',
    'floor_180': 'Turn it another 90 deg counter-clockwise: it faces -x.',
    'floor_270': 'Turn it another 90 deg counter-clockwise: it faces -y.',
    'raised': 'Put it on a stable support 0.5-1 m high (chair, box) near the take-off spot, facing +x, '
              'nothing above it. Do not touch it.' + AWAY,
    'raised_0': 'On the support 0.5-1 m high at the take-off spot, facing +x. Do not touch it.' + AWAY,
    'raised_180': 'Same support and spot, turned to face -x. Do not touch it.' + AWAY,
    'lift': 'Hold it level at the take-off spot, facing +x. After Enter: lift it slowly to ~1 m (5 s), '
            'hold (5 s), lower it to the floor (5 s).',
}
ANGLE_COLUMNS = ['pc_s', 'step', 'bs'] + [f'h{i}' for i in range(N_SENSORS)] + [f'v{i}' for i in range(N_SENSORS)]


# ---- recording --------------------------------------------------------------

class Recorder:
    """Collects angle packets and telemetry rows, tagged with the current step."""

    def __init__(self):
        self.step, self.angles, self.tel, self.lock = None, [], [], threading.Lock()

    def angle_cb(self, packet):
        from cflib.crazyflie.localization import Localization
        if self.step is None or packet.type != Localization.LH_ANGLE_STREAM:
            return
        d = packet.data
        with self.lock:
            self.angles.append([time.monotonic(), self.step, d['basestation']] + list(d['x']) + list(d['y']))

    def tel_cb(self, block):
        def cb(ts, data, conf):
            if self.step is None:
                return
            row = {c: '' for c in TEL_COLUMNS}
            row.update(pc_s=time.monotonic(), step=self.step, block=block)
            for name, _, col in TELEMETRY[block]:
                row[col] = data.get(name, '')
            with self.lock:
                self.tel.append([row[c] for c in TEL_COLUMNS])
        return cb

    def record(self, step, seconds):
        self.step = step
        time.sleep(seconds)
        self.step = None
        n = sum(1 for a in self.angles if a[1] == step)
        log.info('%s: %d angle packets, %d telemetry rows in %.0f s', step, n,
                 sum(1 for r in self.tel if r[1] == step), seconds)


def read_geometry(cf):
    """The drone's stored base-station geometry: {bs: (origin, rotation)}."""
    from cflib.crazyflie.mem import LighthouseMemHelper
    done, geos = threading.Event(), {}
    LighthouseMemHelper(cf).read_all_geos(lambda g: (geos.update(g), done.set()))
    if not done.wait(10):
        raise RuntimeError('could not read the base-station geometry from the drone')
    return {int(k): {'origin': list(map(float, g.origin)), 'rotation': [list(map(float, r)) for r in g.rotation_matrix]}
            for k, g in geos.items() if g.valid}


def record(args):
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.utils import uri_helper
    cflib.crtp.init_drivers()
    uri = uri_helper.uri_from_env(default=DEFAULT_URI)
    if args.steps:
        steps = [(n, STEPS[n]) for n in args.steps]
    else:
        steps = [(n, STEPS[n]) for n in ('floor_0', 'floor_90', 'floor_180', 'floor_270')]
        if args.raised:
            steps.append(('raised', STEPS['raised']))
        if args.lift:
            steps.append(('lift', STEPS['lift']))
    rec = Recorder()
    log.info('Connecting to %s ... (cfclient must be closed)', uri)
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        geometry = read_geometry(cf)
        log.info('geometry on the drone for base stations %s', sorted(geometry))
        toc = cf.log.toc
        confs = []
        for i, block in enumerate(TELEMETRY):
            conf = LogConfig(name=f'Check{i}', period_in_ms=10 if i == 1 else 20)
            for name, kind, _ in block:
                if toc.get_element_by_complete_name(name) is None:
                    log.warning('not in this firmware: %s', name)
                else:
                    conf.add_variable(name, kind)
            cf.log.add_config(conf)
            conf.data_received_cb.add_callback(rec.tel_cb(i))
            conf.start()
            confs.append(conf)
        cf.loc.receivedLocationPacket.add_callback(rec.angle_cb)
        cf.param.set_value('locSrv.enLhAngleStream', 1)
        try:
            for step, instruction in steps:
                log.info('%s  [press Enter]', instruction)
                input()
                time.sleep(1.0)  # let the hand get away
                rec.record(step, args.seconds)
        finally:
            cf.param.set_value('locSrv.enLhAngleStream', 0)
            for conf in confs:
                conf.stop()
    angles_path = run_log.csv_path('angles', script=__file__, when=STARTED_AT)
    run_log.write_csv(angles_path, ANGLE_COLUMNS, rec.angles, script=__file__)
    run_log.write_csv(run_log.csv_path('telemetry', script=__file__, when=STARTED_AT), TEL_COLUMNS, rec.tel,
                      script=__file__)
    geo_path = angles_path.with_name(angles_path.name.replace('_angles_', '_geometry_').replace('.csv', '.json'))
    geo_path.write_text(json.dumps(geometry, indent=1))
    log.info('wrote %s (and the telemetry CSV, the geometry JSON)', angles_path)
    return angles_path


# ---- analysis ---------------------------------------------------------------

def load_csv(path):
    lines = [l for l in open(path) if not l.startswith('#')]
    head = lines[0].strip().split(',')
    rows = [l.rstrip('\n').split(',') for l in lines[1:]]
    return head, rows


def ray(origin, rot, h, v):
    """World ray (origin, unit direction) of V1 angles h, v from a base station."""
    d = np.array([1.0, math.tan(h), math.tan(v)])
    return np.asarray(origin), np.asarray(rot) @ (d / np.linalg.norm(d))


def cross_point(r0, r1):
    """Midpoint of the closest approach of two rays and their miss distance."""
    (p, u), (q, w) = r0, r1
    a, b, c = u @ u, u @ w, w @ w
    d, e = u @ (p - q), w @ (p - q)
    den = a * c - b * b
    s, t = (b * e - c * d) / den, (a * e - b * d) / den
    x0, x1 = p + s * u, q + t * w
    return (x0 + x1) / 2, float(np.linalg.norm(x0 - x1))


def survey_reference():
    """Median still-noise sigma per (quantity, axis) from today's survey, or {}."""
    try:
        meas = json.loads(NOISE_FILE.read_text())['measurements']
    except (OSError, KeyError, ValueError):
        return {}
    ref = {}
    for m in meas:
        if m.get('mode') == 'still':
            ref.setdefault((m['quantity'], m['axis']), []).append(m['sigma'])
    return {k: float(np.median(v)) for k, v in ref.items()}


def analyze(angles_path):
    angles_path = Path(angles_path)
    tel_path = Path(str(angles_path).replace('_angles_', '_telemetry_'))
    geo = {int(k): v for k, v in json.loads(Path(str(angles_path).replace('_angles_', '_geometry_')
                                                 .replace('.csv', '.json')).read_text()).items()}
    _, arows = load_csv(angles_path)
    thead, trows = load_csv(tel_path)
    ti = {c: i for i, c in enumerate(thead)}
    steps = list(dict.fromkeys(r[1] for r in arows + trows))
    stations = sorted(geo)
    ref = survey_reference()
    verdict = {'station': {b: 0 for b in stations}, 'sensor': {i: 0 for i in range(N_SENSORS)}, 'geometry': 0, 'floor': 0,
               'pair': {(b, i): 0 for b in stations for i in range(N_SENSORS)}}

    def tel(step, col):
        v = [float(r[ti[col]]) for r in trows if r[1] == step and r[ti[col]] != '']
        return np.array(v)

    for step in steps:
        A = {b: np.array([[float(x) for x in r[3:]] + [float(r[0])] for r in arows if r[1] == step and int(r[2]) == b])
             for b in stations}
        if not any(len(a) for a in A.values()):
            log.info('== %s: no angle packets', step)
            continue
        dur = max(a[:, -1].max() for a in A.values() if len(a)) - min(a[:, -1].min() for a in A.values() if len(a))
        pos = np.stack([tel(step, c) for c in ('x', 'y', 'z')], 1) if len(tel(step, 'x')) else np.zeros((1, 3))
        centre = pos.mean(0)
        still = step != 'lift'
        log.info('== %s (%.0f s): estimate (%.3f, %.3f, %.3f) m, yaw %.1f deg, spread %s mm', step, dur,
                 *centre, tel(step, 'yaw_deg').mean() if len(tel(step, 'yaw_deg')) else float('nan'),
                 np.round(1000 * pos.std(0), 1))
        rx = tel(step, 'lh_receive').astype(int)
        if len(rx):
            log.info('   firmware received (radio-independent): %s', ', '.join(
                f'bs{b} {100 * np.mean(rx & (1 << b) > 0):.0f}%' for b in stations))
            if still:
                for b in stations:
                    if np.mean(rx & (1 << b) > 0) < 0.8:
                        verdict.setdefault('lost', []).append((step, b, np.mean(rx & (1 << b) > 0)))
        log.info('   bs sensor   rate Hz  stale %  noise h/v urad  = mm at range  outliers')
        row_quality = {}
        for b in stations:
            a = A[b]
            if len(a) < 5:
                log.info('   %d  -- %d packets: station not received', b, len(a))
                row_quality[b] = None
                continue
            dist = float(np.linalg.norm(centre - np.array(geo[b]['origin'])))
            q = []
            for i in range(N_SENSORS):
                h, v = a[:, i], a[:, N_SENSORS + i]
                fresh = np.r_[True, (np.abs(np.diff(h)) > STALE_RAD) | (np.abs(np.diff(v)) > STALE_RAD)]
                hs, vs = h[fresh], v[fresh]
                if still:
                    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-9
                    out = int(np.sum((np.abs(hs - np.median(hs)) > OUTLIER_MAD * mad(hs))
                                     | (np.abs(vs - np.median(vs)) > OUTLIER_MAD * mad(vs))))
                    nh, nv = 1.4826 * np.median(np.abs(hs - np.median(hs))), 1.4826 * np.median(np.abs(vs - np.median(vs)))
                else:  # moving: noise around a 0.1 s moving median is not meaningful here; report rate only
                    out, nh, nv = 0, float('nan'), float('nan')
                rate = len(a) / dur * fresh.mean()
                q.append((rate, 1 - fresh.mean(), nh, nv, out))
                log.info('   %d  %d      %7.1f  %6.1f   %6.1f / %6.1f    = %4.2f / %4.2f    %4d', b, i, rate,
                         100 * (1 - fresh.mean()), 1e6 * nh, 1e6 * nv, 1000 * nh * dist, 1000 * nv * dist, out)
            row_quality[b] = (q, dist)
        # station and photodiode comparison (still steps only)
        ok = [b for b in stations if row_quality.get(b)]
        if still and len(ok) == 2:
            mm = {b: np.array([max(x[2], x[3]) * row_quality[b][1] for x in row_quality[b][0]]) for b in ok}
            rate = {b: np.array([x[0] for x in row_quality[b][0]]) for b in ok}
            for b, o in ((ok[0], ok[1]), (ok[1], ok[0])):
                if np.median(mm[b]) > 2 * np.median(mm[o]) or np.median(rate[b]) < 0.8 * np.median(rate[o]):
                    verdict['station'][b] += 1
            for i in range(N_SENSORS):
                others = [j for j in range(N_SENSORS) if j != i]
                bad = [b for b in ok if mm[b][i] > 2 * np.median(mm[b][others]) or rate[b][i] < 0.8 * np.median(rate[b][others])]
                if len(bad) == len(ok):
                    verdict['sensor'][i] += 1
                elif bad:
                    verdict['pair'][(bad[0], i)] += 1
        # two-station consistency: per photodiode, the rays from both stations
        if len(ok) == 2:
            b0, b1 = ok
            t0, t1 = A[b0][:, -1], A[b1][:, -1]
            pts, miss = [[] for _ in range(N_SENSORS)], [[] for _ in range(N_SENSORS)]
            for k, t in enumerate(t1):
                j = int(np.argmin(np.abs(t0 - t)))
                if abs(t0[j] - t) * 1000 > PAIR_MS:
                    continue
                for i in range(N_SENSORS):
                    p, m = cross_point(ray(geo[b0]['origin'], geo[b0]['rotation'], A[b0][j, i], A[b0][j, N_SENSORS + i]),
                                       ray(geo[b1]['origin'], geo[b1]['rotation'], A[b1][k, i], A[b1][k, N_SENSORS + i]))
                    pts[i].append(p)
                    miss[i].append(m)
            if pts[0]:
                P = np.array([np.median(np.array(p), 0) for p in pts])  # median photodiode positions
                M = np.array([np.median(m) for m in miss])
                c = P.mean(0)
                fwd = P[2:].mean(0) - P[:2].mean(0)
                yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
                d_meas = [np.linalg.norm(P[i] - P[j]) for i in range(4) for j in range(i + 1, 4)]
                d_true = [np.linalg.norm(SENSOR_POS[i] - SENSOR_POS[j]) for i in range(4) for j in range(i + 1, 4)]
                shape = 1000 * max(abs(x - y) for x, y in zip(d_meas, d_true))
                tri = np.array([np.mean(np.array([pts[i][k] for i in range(N_SENSORS)]), 0) for k in range(len(pts[0]))])
                log.info('   two stations: rays miss by %s mm (per photodiode); crossing point (%.3f, %.3f, %.3f) m, '
                         '%.0f mm from the estimate; deck yaw %.1f deg; deck shape error %.1f mm; '
                         'crossing spread %s mm', np.round(1000 * M, 1), *c, 1000 * np.linalg.norm(c - centre), yaw,
                         shape, np.round(1000 * tri.std(0), 1) if still else '(moving)')
                if still and (M.max() > 0.010 or shape > 5):
                    verdict['geometry'] += 1
                if step == 'lift':
                    zr = tel(step, 'zrange_mm')
                    tz = np.array([r[0] for r in arows if r[1] == step])
                    zt = [float(r[0]) for r in trows if r[1] == step and r[ti['zrange_mm']] != '']
                    if len(zr) > 10 and len(tri) > 10:
                        ttri = t1[[k for k, t in enumerate(t1) if abs(t0[int(np.argmin(np.abs(t0 - t)))] - t) * 1000 <= PAIR_MS]]
                        z_lh = np.interp(zt, ttri, tri[:, 2])
                        good = zr < 4000
                        fit = np.polyfit(z_lh[good], zr[good] / 1000, 1)
                        res = zr[good] / 1000 - np.polyval(fit, z_lh[good])
                        log.info('   lift: Lighthouse height %.2f-%.2f m; range = %.3f * LH z %+.3f m, residual %.1f mm; '
                                 'failed range reads %d', z_lh.min(), z_lh.max(), fit[0], fit[1], 1000 * res.std(),
                                 int((~good).sum()))
        if still and step.startswith('floor'):
            verdict['floor'] += 1
        # other sensors at rest
        if still:
            parts = []
            for ax in 'xyz':
                g = tel(step, f'gyro_{ax}') * math.pi / 180
                r = ref.get(('angular_velocity', ax))
                parts.append(f'gyro {ax} mean {math.degrees(g.mean()):+.2f} deg/s sd {g.std():.4f} rad/s'
                             + (f' (survey {r:.4f})' if r else ''))
            log.info('   %s', '; '.join(parts))
            acc = np.stack([tel(step, f'acc_{ax}') for ax in 'xyz'], 1) * 9.81
            log.info('   acc mean (%.2f, %.2f, %.2f) m/s^2, |a| %.3f g, sd %s m/s^2 (survey %s)', *acc.mean(0),
                     np.linalg.norm(acc.mean(0)) / 9.81, np.round(acc.std(0), 3),
                     [round(ref.get(('acceleration', a), float('nan')), 3) for a in 'xyz'])
            zr = tel(step, 'zrange_mm')
            fdx, fdy, sq = tel(step, 'flow_dx'), tel(step, 'flow_dy'), tel(step, 'flow_squal')
            baro, vb = tel(step, 'baro_hpa'), tel(step, 'vbat')
            log.info('   range %.1f mm sd %.2f mm (failed %d); flow dx %.2f sd %.2f, dy %.2f sd %.2f counts, '
                     'quality %.0f; baro sd %.3f hPa; battery %.2f V', np.median(zr), zr[zr < 4000].std(),
                     int((zr >= 4000).sum()), fdx.mean(), fdx.std(), fdy.mean(), fdy.std(), sq.mean(),
                     baro.std(), vb.mean())
            st = tel(step, 'lh_status')
            log.info('   Lighthouse status estimating %.0f%% of samples', 100 * np.mean(st == 2))

    n = max(verdict['floor'], 1)
    log.info('== DIAGNOSIS (over %d floor orientations and the raised step)', verdict['floor'])
    found = False
    for b, k in verdict['station'].items():
        if k >= max(2, n - 1):
            log.info('BASE STATION %d is worse than the other in %d steps on all photodiodes: the station '
                     '(its position, line of sight or the unit itself)', b, k)
            found = True
    for i, k in verdict['sensor'].items():
        if k >= max(2, n - 1):
            log.info('DECK PHOTODIODE %d is worse than the other three for both stations in %d steps: the deck', i, k)
            found = True
    for (b, i), k in verdict['pair'].items():
        if k >= max(2, n - 1):
            log.info('PHOTODIODE %d is worse than the other three for base station %d only, in %d steps: a partly '
                     'damaged or dirty photodiode (it fails at some angles), or a reflection on that station\'s side', i, b, k)
            found = True
    if verdict['geometry']:
        log.info('GEOMETRY: in %d still steps the two stations disagree by more than 10 mm or the deck shape is '
                 'off by more than 5 mm: re-estimate the geometry (or a station moved)', verdict['geometry'])
        found = True
    for step, b, share in verdict.get('lost', []):
        log.info('LOST: base station %d received only %.0f%% of the time in %s (untouched): something '
                 'blocks its line of sight there, or the drone shades itself (compare raised_0 / raised_180)',
                 b, 100 * share, step)
        found = True
    if not found:
        log.info('No station, photodiode or geometry fault found in these steps.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seconds', type=float, default=15, help='seconds per step (default 15)')
    p.add_argument('--no-raised', dest='raised', action='store_false', help='skip the raised step')
    p.add_argument('--no-lift', dest='lift', action='store_false', help='skip the hand lift')
    p.add_argument('--analyze', metavar='ANGLES_CSV', help='only analyse a recording')
    p.add_argument('--steps', nargs='+', choices=list(STEPS), help='record only these steps, in this order')
    args = p.parse_args()
    analyze(args.analyze if args.analyze else record(args))


if __name__ == '__main__':
    main()
