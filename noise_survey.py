"""
Sensor survey: measure, on the real drone, the noise of everything the learned
policy reads, so the simulator's noise model comes from measurements (plan Step 4,
item D3).

Fly it exactly as learned flights fly: Lighthouse and Flow deck, no AI deck (a
heavier drone hovers at higher motor RPM and vibrates differently). Any firmware
works; the hover mode flies the stock PID. Close cfclient first: only one program
can use the radio.

Modes (python noise_survey.py <mode> --help for the options):
  still    drone untouched on the floor, motors off, at several spots (you move it
           when asked). Per spot: noise of every quantity, and the height offset
           from the first spot (put the first spot where the drone takes off).
  heights  drone resting on spacers of known height (default 0, 1, 2, 3, 5, 10 cm).
           Range sensor and estimated height against the true height, including
           below the range sensor's 4 cm minimum (VL53L1X datasheet 7.3).
  hover    stock-PID hovers at several heights. Noise in flight = the value minus
           its centred moving average (5 Hz); gyro and accelerometer noise against
           the motor RPM (the vibration model).
  slide    drone on a spacer, pushed by hand along a straight track of known
           length, flow on and off: estimated distance against the true distance.
  logs     no drone: reads rl_trigger flight logs for Lighthouse jumps and gyro
           saturation.

Every mode writes its raw samples to logs/noise_survey_<mode>_<time>.csv and its
results to the .json next to it, and merges them into l2f/sim/noise_measured.json,
which l2f/sim/train_mission.sh POS_NOISE=1 reads. Results that differ from the
simulator's current values or the datasheets by more than 2x are flagged.

cflib logs at most 100 Hz per log block; the samples here are 100 Hz, split over
five blocks. The radio may not keep up with all five on every setup: the summary
reports how many samples arrived, so check it. The Lighthouse angles are those of
deck sensor 0 only (all the firmware logs), each updated once per rotation (~50 Hz).

Usage: python noise_survey.py still --spots takeoff,centre,landing
       python noise_survey.py heights --heights 0,1,2,3,5,10
       python noise_survey.py hover --hover-heights 0.3,0.6,1.0,1.5
       python noise_survey.py slide --heights 1,3,10 --track 1.0 --flow both
       python noise_survey.py logs [logs/rl_trigger_flight_*.csv ...]
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

import run_log

log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()
ROOT = Path(__file__).resolve().parent
NOISE_FILE = ROOT / 'l2f' / 'sim' / 'noise_measured.json'
LIGHTHOUSE_FILE = ROOT / 'config' / 'current' / 'config.yaml'  # the active room's base stations
DEFAULT_URI = 'radio://0/80/2M/E7E7E7E7EA'
PERIOD_MS = 10            # cflib's shortest log period (100 Hz)
FS = 1000.0 / PERIOD_MS   # samples per second
LOW_VBAT_FLIGHT = 3.2     # V under load for 0.5 s: hover mode lands (the firmware's low-battery level)
BROWNOUT_VBAT = 3.0       # V: samples below it are left out (sensors and motors fail, 2026-11-22)
RANGE_MAX_M = 4.0         # VL53L1X maximum range; failed reads log as 64.35 m
MA_SAMPLES = 21           # centred moving average over 0.2 s: its first zero is at 5 Hz
RANGE_MIN_M = 0.04        # VL53L1X: accurate only from 4 cm (datasheet DS12385, 7.3)
GYRO_SATURATION_DPS = 1990  # at the +-2000 deg/s range (readings can exceed 2000 a little)

# What the simulator uses today (src/config/parameters.h, L2F_REAL_NOISE), in SI units.
SIM_NOISE = {'position': 0.005, 'orientation': 0.006, 'linear_velocity': 0.010,
             'angular_velocity': 0.069}
# BMI088 datasheet gyro noise, 0.014 deg/s/sqrt(Hz), as RMS over 80-230 Hz bandwidth.
GYRO_DATASHEET = (0.0024, 0.0041)

# (log variable, log type, CSV column, factor to SI units, quantity, axis)
VARIABLES = [
    ('stateEstimate.x', 'float', 'x', 1.0, 'position', 'x'),
    ('stateEstimate.y', 'float', 'y', 1.0, 'position', 'y'),
    ('stateEstimate.z', 'float', 'z', 1.0, 'position', 'z'),
    ('stateEstimate.roll', 'float', 'roll', math.pi / 180, 'orientation', 'roll'),
    ('stateEstimate.pitch', 'float', 'pitch', math.pi / 180, 'orientation', 'pitch'),
    ('stateEstimate.yaw', 'float', 'yaw', math.pi / 180, 'orientation', 'yaw'),
    ('stateEstimate.vx', 'float', 'vx', 1.0, 'linear_velocity', 'x'),
    ('stateEstimate.vy', 'float', 'vy', 1.0, 'linear_velocity', 'y'),
    ('stateEstimate.vz', 'float', 'vz', 1.0, 'linear_velocity', 'z'),
    ('range.zrange', 'uint16_t', 'zrange', 0.001, 'range', 'z'),
    ('motion.deltaX', 'int16_t', 'flow_dx', 1.0, 'flow', 'x'),
    ('motion.deltaY', 'int16_t', 'flow_dy', 1.0, 'flow', 'y'),
    ('motion.squal', 'uint8_t', 'flow_squal', 1.0, 'flow_quality', ''),
    ('gyro.x', 'float', 'gyro_x', math.pi / 180, 'angular_velocity', 'x'),
    ('gyro.y', 'float', 'gyro_y', math.pi / 180, 'angular_velocity', 'y'),
    ('gyro.z', 'float', 'gyro_z', math.pi / 180, 'angular_velocity', 'z'),
    ('acc.x', 'float', 'acc_x', 9.81, 'acceleration', 'x'),
    ('acc.y', 'float', 'acc_y', 9.81, 'acceleration', 'y'),
    ('acc.z', 'float', 'acc_z', 9.81, 'acceleration', 'z'),
    ('motor.m1_rpm', 'uint16_t', 'rpm1', 1.0, 'rpm', '1'),
    ('motor.m2_rpm', 'uint16_t', 'rpm2', 1.0, 'rpm', '2'),
    ('motor.m3_rpm', 'uint16_t', 'rpm3', 1.0, 'rpm', '3'),
    ('motor.m4_rpm', 'uint16_t', 'rpm4', 1.0, 'rpm', '4'),
    ('pm.vbat', 'float', 'vbat', 1.0, 'battery', ''),
    ('lighthouse.bsActive', 'uint16_t', 'lh_active', 1.0, 'lighthouse', ''),
    # raw LH2 sweep angles of deck sensor 0 (the firmware logs only that sensor), base
    # stations 0 and 1, sweeps 1 and 2: the angle noise of the simulator's Lighthouse model
    ('lighthouse.rawAngle0xlh2', 'float', 'lh_bs0_sweep0', 1.0, 'lh_angle', 'bs0_sweep0'),
    ('lighthouse.rawAngle0ylh2', 'float', 'lh_bs0_sweep1', 1.0, 'lh_angle', 'bs0_sweep1'),
    ('lighthouse.rawAngle1xlh2', 'float', 'lh_bs1_sweep0', 1.0, 'lh_angle', 'bs1_sweep0'),
    ('lighthouse.rawAngle1ylh2', 'float', 'lh_bs1_sweep1', 1.0, 'lh_angle', 'bs1_sweep1'),
]
# Log blocks of at most 26 bytes each; the first one paces the CSV rows.
BLOCKS = [VARIABLES[0:6], VARIABLES[6:13], VARIABLES[13:19], VARIABLES[19:25], VARIABLES[25:29]]
COLUMNS = ['drone_ms', 'tag'] + [v[2] for v in VARIABLES]
# Quantities whose noise is merged into noise_measured.json and compared.
NOISE_QUANTITIES = ['position', 'orientation', 'linear_velocity', 'angular_velocity', 'acceleration', 'lh_angle']


# ---- statistics -------------------------------------------------------------

def unwrap(values):
    """Remove 2*pi jumps from an angle series (rad)."""
    out, offset = [], 0.0
    for i, v in enumerate(values):
        if i and v + offset - out[-1] > math.pi:
            offset -= 2 * math.pi
        elif i and v + offset - out[-1] < -math.pi:
            offset += 2 * math.pi
        out.append(v + offset)
    return out


def moving_average(values, n=MA_SAMPLES):
    """Centred moving average over n samples (shorter windows at the ends)."""
    half, out = n // 2, []
    for i in range(len(values)):
        window = values[max(0, i - half):i + half + 1]
        out.append(sum(window) / len(window))
    return out


def lag1(values):
    """Lag-1 autocorrelation r1 of a zero-mean series (0 for white noise)."""
    if len(values) < 3:
        return float('nan')
    num = sum(a * b for a, b in zip(values, values[1:]))
    den = sum(a * a for a in values)
    return num / den if den > 0 else float('nan')


def noise(values, detrend):
    """(sigma, r1) of a series: its residual around its mean ('mean') or around its
    centred 5 Hz moving average ('ma', for flight)."""
    if len(values) < 10:
        return float('nan'), float('nan')
    if detrend == 'mean':
        m = sum(values) / len(values)
        residual = [v - m for v in values]
    else:
        residual = [v - a for v, a in zip(values, moving_average(values))]
    return statistics.pstdev(residual), lag1(residual)


def column(rows, name, tag=None):
    """Values of CSV column `name` in SI units (rows with that tag only, Nones dropped)."""
    var = next(v for v in VARIABLES if v[2] == name)
    i = COLUMNS.index(name)
    values = [r[i] * var[3] for r in rows if r[i] is not None and (tag is None or r[1] == tag)]
    if name == 'zrange':
        values = [v for v in values if 0 < v <= RANGE_MAX_M]
    return unwrap(values) if var[4] == 'orientation' else values


def describe(rows, tag, detrend):
    """Per variable: mean, sigma and r1 (SI units) for the rows with `tag`."""
    out = {}
    for var in VARIABLES:
        values = column(rows, var[2], tag)
        if not values:
            continue
        sigma, r1 = noise(values, detrend)
        out[var[2]] = {'quantity': var[4], 'axis': var[5], 'mean': sum(values) / len(values),
                       'sigma': sigma, 'r1': r1, 'samples': len(values)}
    return out


# ---- noise_measured.json ----------------------------------------------------

def placement():
    """The active base-station placement: room folder and a short fingerprint of its
    config.yaml. Sensor-level results (IMU, range, flow, sweep-angle noise) hold for any
    placement; estimate-level ones (position noise, Lighthouse jumps) and the replay check
    belong to the placement they were measured with."""
    if not LIGHTHOUSE_FILE.exists():
        return {'room': None, 'sha256': None}
    return {'room': LIGHTHOUSE_FILE.resolve().parent.name,
            'sha256': hashlib.sha256(LIGHTHOUSE_FILE.read_bytes()).hexdigest()[:12]}



def merge(mode, where_to_stats, extra=None):
    """Add this run's measurements to l2f/sim/noise_measured.json and recompute each
    quantity's sigma range per axis (min and max over every measurement so far)."""
    data = json.loads(NOISE_FILE.read_text()) if NOISE_FILE.exists() else {}
    entries = data.setdefault('measurements', [])
    stamp = STARTED_AT.isoformat(timespec='seconds')
    for where, stats in where_to_stats.items():
        for name, s in stats.items():
            if s['quantity'] in NOISE_QUANTITIES and not math.isnan(s['sigma']):
                entries.append({'mode': mode, 'where': where, 'quantity': s['quantity'],
                                'axis': s['axis'], 'sigma': s['sigma'], 'r1': s['r1'], 'time': stamp,
                                'placement': placement()['sha256']})
    for quantity in NOISE_QUANTITIES:
        for axis in sorted({e['axis'] for e in entries if e['quantity'] == quantity}):
            sel = [e for e in entries if e['quantity'] == quantity and e['axis'] == axis]
            r1 = [e['r1'] for e in sel if not math.isnan(e['r1'])]
            data.setdefault(quantity, {})[axis] = {
                'sigma_min': min(e['sigma'] for e in sel), 'sigma_max': max(e['sigma'] for e in sel),
                'r1_median': statistics.median(r1) if r1 else None, 'measurements': len(sel)}
    for key, value in (extra or {}).items():
        data[key] = value
    data['updated'] = stamp
    NOISE_FILE.parent.mkdir(parents=True, exist_ok=True)
    NOISE_FILE.write_text(json.dumps(data, indent=1, sort_keys=True) + '\n')
    log.info('Merged into %s', NOISE_FILE)


def compare(where_to_stats):
    """Flag noise more than 2x away from what the simulator uses, and the gyro against
    the datasheet (in flight, vibration should make it much larger)."""
    for where, stats in where_to_stats.items():
        for name, s in stats.items():
            ref = SIM_NOISE.get(s['quantity'])
            if ref and not math.isnan(s['sigma']) and not 0.5 <= s['sigma'] / ref <= 2:
                log.warning('%s %s: sigma %.4g is %.1fx the simulator\'s %.4g (more than 2x apart: '
                            'explain before using it)', where, name, s['sigma'], s['sigma'] / ref, ref)
            if s['quantity'] == 'angular_velocity' and not math.isnan(s['sigma']):
                log.info('%s %s: sigma %.4f rad/s = %.0fx the BMI088 datasheet electronic noise '
                         '(%.4f-%.4f rad/s)', where, name, s['sigma'], s['sigma'] / GYRO_DATASHEET[1],
                         *GYRO_DATASHEET)


def report(where_to_stats):
    """Log the sigma of the main quantities per place."""
    for where, stats in where_to_stats.items():
        parts = [f"{n} {s['sigma']:.4g}" for n, s in stats.items() if s['quantity'] in NOISE_QUANTITIES + ['range']]
        log.info('%s: sigma (SI) %s', where, ', '.join(parts))


def save(mode, rows, results):
    """Raw samples to logs/noise_survey_<mode>_<time>.csv, results to the .json next to it."""
    path = run_log.csv_path(mode, script=__file__, when=STARTED_AT)
    run_log.write_csv(path, COLUMNS, rows, script=__file__, generated_at=datetime.now().astimezone())
    json_path = path.with_suffix('.json')
    results = dict(results, lighthouse_placement=placement())
    json_path.write_text(json.dumps(results, indent=1, sort_keys=True) + '\n')
    if LIGHTHOUSE_FILE.exists():  # the geometry these samples were taken with (replay check)
        path.with_name(path.stem + '_lighthouse.yaml').write_bytes(LIGHTHOUSE_FILE.read_bytes())
    log.info('Samples: %s (%d rows); results: %s; base stations: %s', path, len(rows), json_path,
             results['lighthouse_placement'])


# ---- drone ----------------------------------------------------------------------

class Recorder:
    """Rows at the first log block's rate, with the newest values of the other blocks.
    Only records while `tag` is set."""
    def __init__(self):
        self.latest = {}
        self.rows = []
        self.tag = None
        self.stop = threading.Event()  # set to end a recording early (low battery)

    def first_cb(self, timestamp, data, logconf):
        """Pace the rows: one per sample of the first block."""
        self.latest.update(data)
        if self.tag is not None:
            self.rows.append([timestamp, self.tag] + [self.latest.get(v[0]) for v in VARIABLES])

    def other_cb(self, timestamp, data, logconf):
        """Remember the newest values of the other blocks."""
        self.latest.update(data)

    def record(self, tag, seconds):
        """Record `seconds` of samples under `tag`; return how many arrived."""
        before = len(self.rows)
        self.tag = tag
        self.stop.wait(seconds)
        self.tag = None
        got = len(self.rows) - before
        log.info('%s: %d samples in %.0f s (%.0f%% of %d expected)', tag, got, seconds,
                 100 * got / (seconds * FS), seconds * FS)
        return got


def start_logging(scf, rec):
    """Start the 100 Hz log blocks with every variable this firmware has."""
    from cflib.crazyflie.log import LogConfig
    toc = scf.cf.log.toc
    confs, missing = [], []
    for i, block in enumerate(BLOCKS):
        conf = LogConfig(name=f'Survey{i}', period_in_ms=PERIOD_MS)
        for name, kind, *_ in block:
            if toc.get_element_by_complete_name(name) is None:
                missing.append(name)
            else:
                conf.add_variable(name, kind)
        if conf.variables:
            scf.cf.log.add_config(conf)
            conf.data_received_cb.add_callback(rec.first_cb if i == 0 else rec.other_cb)
            conf.start()
            confs.append(conf)
    if missing:
        log.warning('Not in this firmware (not logged): %s', ', '.join(missing))
    return confs


def connect(args):
    """Open the radio link; returns the SyncCrazyflie (use it as a context manager)."""
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.utils import uri_helper
    cflib.crtp.init_drivers()
    uri = uri_helper.uri_from_env(default=DEFAULT_URI)
    log.info('Connecting to %s ... (cfclient must be closed)', uri)
    return SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache'))


def ask(message):
    """Log an instruction and wait for Enter."""
    log.info('%s  [press Enter]', message)
    input()


def run_still(args):
    """Drone untouched on the floor at each spot: noise and height offsets."""
    from cflib.utils.reset_estimator import reset_estimator
    rec = Recorder()
    with connect(args) as scf:
        confs = start_logging(scf, rec)
        for spot in args.spots:
            ask(f'Put the drone at spot "{spot}" (motors off, do not touch it).')
            reset_estimator(scf)
            time.sleep(1.0)
            rec.record(spot, args.seconds)
        for conf in confs:
            conf.stop()
    stats = {spot: describe(rec.rows, spot, 'mean') for spot in args.spots}
    z0 = stats[args.spots[0]].get('z', {}).get('mean')
    offsets = {spot: s['z']['mean'] - z0 for spot, s in stats.items() if 'z' in s and z0 is not None}
    for spot, dz in offsets.items():
        log.info('Height offset of "%s" from "%s": %+.1f mm', spot, args.spots[0], 1000 * dz)
    report(stats)
    compare(stats)
    results = {'mode': 'still', 'stats': stats, 'height_offset_m': offsets}
    save('still', rec.rows, results)
    merge('still', stats, {'still_height_offset_m': offsets})


def run_heights(args):
    """Drone resting on spacers: range sensor and estimated height against the truth."""
    from cflib.utils.reset_estimator import reset_estimator
    rec = Recorder()
    with connect(args) as scf:
        confs = start_logging(scf, rec)
        for h in args.heights:
            ask(f'Put the drone on the {h:g} cm spacer (0 = on the floor), motors off.')
            reset_estimator(scf)
            time.sleep(1.0)
            rec.record(f'{h:g}cm', args.seconds)
        for conf in confs:
            conf.stop()
    table = []
    for h in args.heights:
        tag = f'{h:g}cm'
        d, z = column(rec.rows, 'zrange', tag), column(rec.rows, 'z', tag)
        if not d or not z:
            continue
        table.append({'height_m': h / 100, 'range_mean_m': sum(d) / len(d), 'range_sigma_m': noise(d, 'mean')[0],
                      'z_mean_m': sum(z) / len(z), 'z_sigma_m': noise(z, 'mean')[0]})
    if table:
        base = table[0]
        for row in table:
            row['z_bias_m'] = (row['z_mean_m'] - base['z_mean_m']) - (row['height_m'] - base['height_m'])
            row['below_range_minimum'] = row['range_mean_m'] < RANGE_MIN_M
            log.info('%4.0f cm: range %.1f +- %.1f mm%s, estimated height %+.1f mm vs true (sigma %.1f mm)',
                     100 * row['height_m'], 1000 * row['range_mean_m'], 1000 * row['range_sigma_m'],
                     ' (below the 4 cm minimum)' if row['below_range_minimum'] else '',
                     1000 * row['z_bias_m'], 1000 * row['z_sigma_m'])
        log.info('Range sensor distance from the floor when landed: %.1f mm', 1000 * base['range_mean_m'])
    stats = {f'{h:g}cm': describe(rec.rows, f'{h:g}cm', 'mean') for h in args.heights}
    results = {'mode': 'heights', 'table': table, 'stats': stats}
    save('heights', rec.rows, results)
    merge('heights', stats, {'range_vs_height': table})


def run_hover(args):
    """Stock-PID hovers at several heights: noise in flight and vibration against RPM."""
    from cflib.utils.reset_estimator import reset_estimator
    rec = Recorder()
    with connect(args) as scf:
        cf = scf.cf
        if cf.param.get_value('stabilizer.controller') != '1':
            log.info('Selecting the PID controller')
            cf.param.set_value('stabilizer.controller', '1')
            time.sleep(0.5)
        reset_estimator(scf)
        confs = start_logging(scf, rec)
        time.sleep(1.0)
        x0, y0, z0 = (rec.latest.get(f'stateEstimate.{a}') for a in 'xyz')
        vbat = rec.latest.get('pm.vbat')
        if None in (x0, y0, z0) or (vbat is not None and vbat < args.min_vbat):
            log.error('No position estimate, or battery %s V below %.2f V: not flying.', vbat, args.min_vbat)
            for conf in confs:
                conf.stop()
            return
        top = max(args.hover_heights)
        ask(f'Hovers at {", ".join(f"{h:g}" for h in args.hover_heights)} m above this spot, '
            f'{args.seconds:g} s each; clear space up to {top:g} m. Ctrl+C cuts the motors.')
        # Emergency-stop watchdog: the firmware cuts the motors if these keep-alives
        # stop for 1 s (script crash, frozen laptop, lost radio).
        alive = threading.Event()
        alive.set()

        def keep_alive():
            while alive.is_set():
                cf.supervisor.send_emergency_stop_watchdog()
                time.sleep(0.2)

        threading.Thread(target=keep_alive, daemon=True).start()
        low = threading.Event()

        def watch_battery():
            # land instead of flying into a brown-out: the voltage under load below
            # LOW_VBAT_FLIGHT for 0.5 s ends the recording and skips the other heights
            since = None
            while alive.is_set() and not low.is_set():
                v = rec.latest.get('pm.vbat')
                if v is not None and v < LOW_VBAT_FLIGHT:
                    since = since or time.time()
                    if time.time() - since > 0.5:
                        log.warning('Battery %.2f V under load: landing, the other heights are skipped.', v)
                        low.set()
                        rec.stop.set()
                else:
                    since = None
                time.sleep(0.05)

        threading.Thread(target=watch_battery, daemon=True).start()
        hlc = cf.high_level_commander
        cf.supervisor.send_arming_request(True)
        time.sleep(1.0)
        try:
            hlc.takeoff(z0 + args.hover_heights[0], 2.5)
            time.sleep(3.0)
            for h in args.hover_heights:
                if low.is_set():
                    break
                hlc.go_to(x0, y0, z0 + h, 0.0, 2.0)
                if not low.wait(3.0):  # settle
                    rec.record(f'{h:g}m', args.seconds)
            hlc.land(z0, 2.5)
            time.sleep(3.0)
            hlc.stop()
        except KeyboardInterrupt:
            log.warning('Ctrl+C: EMERGENCY STOP, cutting motors.')
            cf.supervisor.send_emergency_stop()
        finally:
            alive.clear()
            cf.commander.send_stop_setpoint()
            cf.supervisor.send_arming_request(False)
            for conf in confs:
                conf.stop()
    vi = COLUMNS.index('vbat')
    rows = [r for r in rec.rows if r[vi] is None or r[vi] >= BROWNOUT_VBAT]
    if len(rows) < len(rec.rows):
        log.warning('%d samples below %.1f V left out of the statistics.', len(rec.rows) - len(rows), BROWNOUT_VBAT)
    tags = [f'{h:g}m' for h in args.hover_heights]
    stats = {tag: describe(rows, tag, 'ma') for tag in tags}
    stats = {tag: st for tag, st in stats.items() if st}  # heights not flown (battery)
    vibration = []
    for h, tag in zip(args.hover_heights, tags):
        rpm = [v for c in ('rpm1', 'rpm2', 'rpm3', 'rpm4') for v in column(rows, c, tag) if v < 65535]
        if not rpm or tag not in stats:
            continue
        s = stats[tag]
        vibration.append({'height_m': h, 'rpm_mean': sum(rpm) / len(rpm),
                          'gyro_sigma_rad_s': [s.get(f'gyro_{a}', {}).get('sigma') for a in 'xyz'],
                          'acc_sigma_m_s2': [s.get(f'acc_{a}', {}).get('sigma') for a in 'xyz']})
        log.info('%s: mean motor speed %.0f rpm, gyro sigma %s rad/s', tag, vibration[-1]['rpm_mean'],
                 ', '.join('%.4f' % g for g in vibration[-1]['gyro_sigma_rad_s'] if g is not None))
    report(stats)
    compare(stats)
    results = {'mode': 'hover', 'stats': stats, 'vibration_vs_rpm': vibration}
    save('hover', rec.rows, results)
    merge('hover', stats, {'vibration_vs_rpm': vibration})


def run_slide(args):
    """Drone on a spacer pushed along a track: estimated against true distance."""
    rec = Recorder()
    flows = {'on': [0], 'off': [1], 'both': [0, 1]}[args.flow]
    table = []
    with connect(args) as scf:
        cf = scf.cf
        confs = start_logging(scf, rec)
        for h in args.heights:
            for disable in flows:
                cf.param.set_value('motion.disable', str(disable))
                time.sleep(0.3)
                tag = f'{h:g}cm_flow_{"off" if disable else "on"}'
                ask(f'{tag}: put the drone on the {h:g} cm spacer at the start of the {args.track:g} m track. '
                    'Press Enter, push it steadily to the end, then press Enter again.')
                rec.tag = tag
                start = time.time()
                input()
                rec.tag = None
                duration = time.time() - start
                rows = [r for r in rec.rows if r[1] == tag]
                if len(rows) < 20:
                    log.warning('%s: only %d samples, skipped', tag, len(rows))
                    continue
                vx, vy = column(rows, 'vx'), column(rows, 'vy')
                dt = 1 / FS
                travelled = sum(math.hypot(a, b) * dt for a, b in zip(vx, vy))
                xs, ys = column(rows, 'x'), column(rows, 'y')
                n = max(1, int(0.2 * FS))
                moved = math.hypot(sum(xs[-n:]) / n - sum(xs[:n]) / n, sum(ys[-n:]) / n - sum(ys[:n]) / n)
                row = {'height_m': h / 100, 'flow': 'off' if disable else 'on', 'seconds': duration,
                       'track_m': args.track, 'lighthouse_moved_m': moved, 'velocity_integral_m': travelled,
                       'g_track': travelled / args.track, 'g_lighthouse': travelled / moved if moved > 0 else None}
                table.append(row)
                log.info('%s: integrated estimated speed %.3f m over a %.3f m track (g = %.2f), '
                         'Lighthouse moved %.3f m', tag, travelled, args.track, row['g_track'], moved)
        cf.param.set_value('motion.disable', '0')
        for conf in confs:
            conf.stop()
    results = {'mode': 'slide', 'table': table}
    save('slide', rec.rows, results)
    merge('slide', {}, {'flow_slide': table})


def read_flight_log(path):
    """rl_trigger flight CSV as a list of dicts (comment lines skipped, numbers parsed)."""
    with open(path, newline='') as f:
        lines = [line for line in f if not line.startswith('#')]
    out = []
    for row in csv.DictReader(lines):
        parsed = {}
        for k, v in row.items():
            try:
                parsed[k] = float(v)
            except (TypeError, ValueError):
                parsed[k] = v
        out.append(parsed)
    return out


def pair_jumps(jumps, max_duration=1.0):
    """Merge each jump with the jump back that undoes it (within max_duration s, the two
    displacements cancelling to within half the size) into one event with a duration.
    Jumps without a return keep duration None."""
    events, used = [], set()
    for i, a in enumerate(jumps):
        if i in used:
            continue
        a = dict(a, duration_s=None)
        for j in range(i + 1, len(jumps)):
            b = jumps[j]
            if b['file'] != a['file'] or b['t_s'] - a['t_s'] > max_duration:
                break
            rest = math.sqrt((a['dx'] + b['dx']) ** 2 + (a['dy'] + b['dy']) ** 2 + (a['dz'] + b['dz']) ** 2)
            if j not in used and rest < 0.5 * a['size_m']:
                a['duration_s'] = b['t_s'] - a['t_s']
                used.add(j)
                break
        events.append(a)
    return events


def run_logs(args):
    """Lighthouse jumps and gyro saturation in rl_trigger flight logs."""
    paths = args.files or sorted(glob.glob(str(ROOT / 'logs' / 'rl_trigger_flight_*.csv')))
    if not paths:
        log.error('No rl_trigger flight logs found (give the files, e.g. copied from the flying machine).')
        return
    jumps, minutes, saturated, gyro_samples = [], 0.0, 0, 0
    for path in paths:
        rows = [r for r in read_flight_log(path)
                if all(isinstance(r.get(k), float) for k in ('drone_ms', 'x', 'y', 'z'))]
        if len(rows) < 2:
            continue
        minutes += (rows[-1]['drone_ms'] - rows[0]['drone_ms']) / 60000
        for a, b in zip(rows, rows[1:]):
            dt = (b['drone_ms'] - a['drone_ms']) / 1000
            d = [b[k] - a[k] for k in 'xyz']
            size = math.sqrt(sum(c * c for c in d))
            # a jump: the estimate moves further in one sample than the drone can (2 m/s)
            if dt > 0 and size > max(args.jump, 2.0 * dt):
                jumps.append({'file': Path(path).name, 't_s': a['drone_ms'] / 1000, 'size_m': size,
                              'z_share': abs(d[2]) / size, 'dx': d[0], 'dy': d[1], 'dz': d[2]})
        for r in rows:
            g = [r.get(k) for k in ('gyro_x', 'gyro_y', 'gyro_z')]
            if all(isinstance(v, float) for v in g):
                gyro_samples += 1
                saturated += max(abs(v) for v in g) >= GYRO_SATURATION_DPS
    jumps = pair_jumps(jumps)
    k_z = statistics.mean(j['z_share'] for j in jumps) if jumps else None
    log.info('%d logs, %.1f min of data: %d position jumps (> %.0f cm in one sample), %.2f per minute%s',
             len(paths), minutes, len(jumps), 100 * args.jump, len(jumps) / minutes if minutes else 0,
             f', mean z share {k_z:.2f}' if jumps else '')
    if jumps:
        sizes = sorted(j['size_m'] for j in jumps)
        durations = sorted(j['duration_s'] for j in jumps if j['duration_s'] is not None)
        log.info('Jump size: median %.1f cm, max %.1f cm; %d with a return, duration median %s s',
                 100 * statistics.median(sizes), 100 * sizes[-1], len(durations),
                 '%.2f' % statistics.median(durations) if durations else '-')
    log.info('Gyro: %d of %d samples at the +-2000 deg/s range', saturated, gyro_samples)
    # the flight logs' own placement is not recorded in them: these jumps belong to whatever
    # base-station placement was active when they were flown
    summary = {'files': len(paths), 'minutes': minutes, 'jumps': len(jumps),
               'jumps_per_minute': len(jumps) / minutes if minutes else None,
               'z_share_mean': k_z, 'size_m': sorted(j['size_m'] for j in jumps),
               'duration_s': sorted(j['duration_s'] for j in jumps if j['duration_s'] is not None),
               'gyro_saturated_samples': saturated, 'gyro_samples': gyro_samples}
    path = run_log.csv_path('logs', script=__file__, when=STARTED_AT)
    fields = ['file', 't_s', 'size_m', 'z_share', 'dx', 'dy', 'dz', 'duration_s']
    run_log.write_csv(path, fields, [[j[k] for k in fields] for j in jumps],
                      script=__file__, generated_at=datetime.now().astimezone())
    path.with_suffix('.json').write_text(json.dumps(summary, indent=1) + '\n')
    log.info('Jumps: %s; summary: %s', path, path.with_suffix('.json'))
    merge('logs', {}, {'lighthouse_jumps': summary})


def numbers(text):
    """Comma-separated numbers."""
    return [float(x) for x in text.split(',') if x.strip()]


def main():
    """Parse the mode and its options and run it."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('still', help='drone untouched on the floor at several spots')
    p.add_argument('--spots', type=lambda s: [x for x in s.split(',') if x], default=['takeoff', 'centre', 'landing'],
                   help='spot names, first = the take-off spot (default takeoff,centre,landing)')
    p.add_argument('--seconds', type=float, default=20, help='seconds per spot (default 20)')
    p.set_defaults(func=run_still)
    p = sub.add_parser('heights', help='drone resting on spacers of known height')
    p.add_argument('--heights', type=numbers, default=[0, 1, 2, 3, 5, 10], help='spacer heights in cm, first 0')
    p.add_argument('--seconds', type=float, default=15, help='seconds per spacer (default 15)')
    p.set_defaults(func=run_heights)
    p = sub.add_parser('hover', help='stock-PID hovers at several heights')
    p.add_argument('--hover-heights', type=numbers, default=[0.3, 0.6, 1.0, 1.5], help='m above the take-off spot')
    p.add_argument('--seconds', type=float, default=15, help='seconds per height (default 15)')
    p.add_argument('--min-vbat', type=float, default=3.75, help='refuse to fly below this battery voltage')
    p.set_defaults(func=run_hover)
    p = sub.add_parser('slide', help='drone pushed along a track on a spacer')
    p.add_argument('--heights', type=numbers, default=[1, 3, 10], help='spacer heights in cm')
    p.add_argument('--track', type=float, default=1.0, help='track length in m (default 1.0)')
    p.add_argument('--flow', choices=['on', 'off', 'both'], default='both', help='optical flow on, off or both')
    p.set_defaults(func=run_slide)
    p = sub.add_parser('logs', help='Lighthouse jumps and gyro saturation in rl_trigger logs')
    p.add_argument('files', nargs='*', help='rl_trigger flight CSVs (default: logs/rl_trigger_flight_*.csv)')
    p.add_argument('--jump', type=float, default=0.05, help='smallest jump counted, m per sample (default 0.05)')
    p.set_defaults(func=run_logs)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
