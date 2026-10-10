"""
Crazyflie show: figure 8 -> backflip -> side flip (optional) -> land.

Built on flip.py v5 (same flip tuning) and the figure-8 trajectory from
autonomous_sequence_hl.py.

    python show.py                  # figure 8, backflip, side flip, land
    python show.py --no-side-flip   # figure 8, backflip, land
    python show.py --no-flips       # figure 8 only (test this FIRST)

Or set DO_SIDE_FLIP = False below to turn the side flip off permanently.

Space: the figure 8 swings about +-0.95 m in x and +-0.5 m in y around the
takeoff point, and each flip drifts ~0.5 m. Keep ~3 x 2 m clear and stay
inside Lighthouse coverage. Fully charged battery, hand on Ctrl+C.

Ctrl+C once  -> tries to land.  Ctrl+C twice -> cuts the motors.

Each run writes logs/show_YYYYMMDD_HHMMSS.csv (10 ms samples, with phase).
UNTESTED as a combined sequence: run --no-flips first.
"""
import argparse
import csv
import logging
import os
import sys
import threading
import time
from datetime import datetime

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem import Poly4D
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.utils.reset_estimator import reset_estimator

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('show')
LOG_FILE = os.path.join('logs', datetime.now().strftime('show_%Y%m%d_%H%M%S.csv'))

# ---- What to fly -----------------------------------------------------------
DO_BACKFLIP = True
DO_SIDE_FLIP = True          # <- set False to skip the side flip
BACKFLIP_DIRECTION = 1       # -1 = forward flip instead
SIDE_FLIP_DIRECTION = 1      # -1 = roll the other way

# ---- Height ----------------------------------------------------------------
CLIMB_ABOVE_START = 1.0      # m above the takeoff point
MAX_Z = 2.0                  # m, hard cap: keep below your ceiling
TAKEOFF_TIME = 2.5           # s

# ---- Figure 8 --------------------------------------------------------------
FIG8_TIME_SCALE = 1.0        # >1 = slower (1.5 = 50 % slower)
TRAJ_ID = 1

# ---- Flip tuning (same as flip.py v5) --------------------------------------
ENTRY_VZ = 1.0               # m/s upward speed before rotating
BOOST_FACTOR = 1.7           # x hover thrust while boosting
BOOST_MAX_TIME = 0.45        # s
FLIP_RATE = 1000             # deg/s commanded rotation rate
MAX_FLIP_TIME = 1.0          # s, safety limit for the rotation
UP_FACTOR = 1.5              # x hover thrust when roughly upright
INVERTED_FACTOR = 0.45       # x hover thrust when inverted
EARLY_UP_END = 50            # deg: high thrust until here...
UP_AGAIN_ANGLE = 245         # deg: ...and again from here
BRAKE_ANGLE = 235            # deg: start slowing the rotation
BRAKE_RATE = 250             # deg/s while braking
HANDOVER_MIN = 290           # deg: earliest hand-over to angle mode
HANDOVER_GYRO = 350          # deg/s: hand over once slower than this...
HANDOVER_MAX = 345           # deg: ...or at this angle regardless
MAX_THRUST = 60000
MIN_RECOVER_TIME = 0.15      # s at full power, at least
MAX_RECOVER_TIME = 0.8       # s at full power, at most

# ---- Between flips ---------------------------------------------------------
CATCH_TIME = 1.5             # s of position setpoints right after a flip
SETTLE_TIME = 3.0            # s hovering before the next flip
SETTLE_MAX_POS_ERR = 0.35    # m: skip the next flip if further off than this
SETTLE_MAX_VZ = 0.25         # m/s: ...or still moving vertically faster

# ---- Battery ---------------------------------------------------------------
MIN_BATTERY_V = 3.5          # on the ground, before takeoff
MIN_BATTERY_LOADED_V = 3.3   # in the air (voltage sags under load)
FALLBACK_HOVER_THRUST = 35000
RATE_HZ = 100
# ----------------------------------------------------------------------------

DT = 1.0 / RATE_HZ

AXES = {
    'pitch': {'param': 'flightmode.stabModePitch'},
    'roll': {'param': 'flightmode.stabModeRoll'},
}

# Duration, x^0..x^7, y^0..y^7, z^0..z^7, yaw^0..yaw^7 (from autonomous_sequence_hl.py)
FIGURE8 = [
    [1.050000, 0.000000, -0.000000, 0.000000, -0.000000, 0.830443, -0.276140, -0.384219, 0.180493, -0.000000, 0.000000, -0.000000, 0.000000, -1.356107, 0.688430, 0.587426, -0.329106, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [0.710000, 0.396058, 0.918033, 0.128965, -0.773546, 0.339704, 0.034310, -0.026417, -0.030049, -0.445604, -0.684403, 0.888433, 1.493630, -1.361618, -0.139316, 0.158875, 0.095799, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [0.620000, 0.922409, 0.405715, -0.582968, -0.092188, -0.114670, 0.101046, 0.075834, -0.037926, -0.291165, 0.967514, 0.421451, -1.086348, 0.545211, 0.030109, -0.050046, -0.068177, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [0.700000, 0.923174, -0.431533, -0.682975, 0.177173, 0.319468, -0.043852, -0.111269, 0.023166, 0.289869, 0.724722, -0.512011, -0.209623, -0.218710, 0.108797, 0.128756, -0.055461, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [0.560000, 0.405364, -0.834716, 0.158939, 0.288175, -0.373738, -0.054995, 0.036090, 0.078627, 0.450742, -0.385534, -0.954089, 0.128288, 0.442620, 0.055630, -0.060142, -0.076163, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [0.560000, 0.001062, -0.646270, -0.012560, -0.324065, 0.125327, 0.119738, 0.034567, -0.063130, 0.001593, -1.031457, 0.015159, 0.820816, -0.152665, -0.130729, -0.045679, 0.080444, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [0.700000, -0.402804, -0.820508, -0.132914, 0.236278, 0.235164, -0.053551, -0.088687, 0.031253, -0.449354, -0.411507, 0.902946, 0.185335, -0.239125, -0.041696, 0.016857, 0.016709, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [0.620000, -0.921641, -0.464596, 0.661875, 0.286582, -0.228921, -0.051987, 0.004669, 0.038463, -0.292459, 0.777682, 0.565788, -0.432472, -0.060568, -0.082048, -0.009439, 0.041158, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [0.710000, -0.923935, 0.447832, 0.627381, -0.259808, -0.042325, -0.032258, 0.001420, 0.005294, 0.288570, 0.873350, -0.515586, -0.730207, -0.026023, 0.288755, 0.215678, -0.148061, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [1.053185, -0.398611, 0.850510, -0.144007, -0.485368, -0.079781, 0.176330, 0.234482, -0.153567, 0.447039, -0.532729, -0.855023, 0.878509, 0.775168, -0.391051, -0.713519, 0.391628, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
]

CSV_HEADER = ['drone_ms', 'phase', 'x', 'y', 'z', 'vz', 'pitch_deg', 'roll_deg',
              'gyro_x_dps', 'gyro_y_dps', 'rotation_deg', 'cmd_thrust', 'vbat',
              'sent_roll_rate', 'sent_pitch_rate', 'sent_thrust']


class Telemetry:
    """Log streams: state + gyro at 10 ms, thrust/battery/supervisor at 20 ms."""

    def __init__(self):
        self.lock = threading.Lock()
        self.x = self.y = self.z = self.vz = 0.0
        self.pitch = self.roll = 0.0
        self.gx = self.gy = 0.0
        self.thrust = 0.0
        self.vbat = None
        self.sup = None
        self.sent = (0.0, 0.0, 0.0)
        self.axis = 'pitch'
        self.last_gyro_ts = None
        self.rotation = self.rot_max = 0.0
        self.integrating = False
        self.tracking = False
        self.min_z = self.max_z = 0.0
        self.phase = 'ground'
        self.rows = []
        self.received = False

    def state_cb(self, ts, d, lc):
        with self.lock:
            self.x, self.y = d['stateEstimate.x'], d['stateEstimate.y']
            self.z, self.vz = d['stateEstimate.z'], d['stateEstimate.vz']
            self.pitch, self.roll = d['stabilizer.pitch'], d['stabilizer.roll']
            if self.tracking:
                self.min_z = min(self.min_z, self.z)
                self.max_z = max(self.max_z, self.z)
            self.received = True
            self.rows.append([ts, self.phase, self.x, self.y, self.z, self.vz,
                              self.pitch, self.roll, self.gx, self.gy, self.rotation,
                              self.thrust, self.vbat, *self.sent])

    def gyro_cb(self, ts, d, lc):
        with self.lock:
            self.gx, self.gy = d['gyro.x'], d['gyro.y']
            if self.integrating and self.last_gyro_ts is not None:
                rate = self.gy if self.axis == 'pitch' else self.gx
                self.rotation += rate * (ts - self.last_gyro_ts) / 1000.0
                if abs(self.rotation) > abs(self.rot_max):
                    self.rot_max = self.rotation
            self.last_gyro_ts = ts

    def slow_cb(self, ts, d, lc):
        with self.lock:
            self.thrust = d['controller.cmd_thrust']
            self.vbat = d['pm.vbat']
            self.sup = int(d['supervisor.info'])

    def start_rotation(self, axis):
        with self.lock:
            self.axis = axis
            self.rotation = self.rot_max = 0.0
            self.integrating = True

    def stop_rotation(self):
        with self.lock:
            self.integrating = False

    def start_tracking(self):
        with self.lock:
            self.min_z = self.max_z = self.z
            self.tracking = True

    def stop_tracking(self):
        with self.lock:
            self.tracking = False

    def snapshot(self, axis):
        """z, vz, attitude angle on axis, gyro rate on axis, integrated rotation."""
        with self.lock:
            if axis == 'pitch':
                return self.z, self.vz, self.pitch, self.gy, self.rotation
            return self.z, self.vz, self.roll, self.gx, self.rotation

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with self.lock:
            rows = list(self.rows)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            w.writerows(rows)
        log.info('Flight log written to %s (%d samples)', path, len(rows))


def start_telemetry(scf, tele):
    confs = []
    state = LogConfig(name='ShowState', period_in_ms=10)
    for name in ('stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
                 'stateEstimate.vz', 'stabilizer.pitch', 'stabilizer.roll'):
        state.add_variable(name, 'float')
    state.data_received_cb.add_callback(tele.state_cb)
    confs.append(state)

    gyro = LogConfig(name='ShowGyro', period_in_ms=10)
    gyro.add_variable('gyro.x', 'float')
    gyro.add_variable('gyro.y', 'float')
    gyro.data_received_cb.add_callback(tele.gyro_cb)
    confs.append(gyro)

    slow = LogConfig(name='ShowSlow', period_in_ms=20)
    slow.add_variable('controller.cmd_thrust', 'float')
    slow.add_variable('pm.vbat', 'float')
    slow.add_variable('supervisor.info', 'uint16_t')
    slow.data_received_cb.add_callback(tele.slow_cb)
    confs.append(slow)

    for c in confs:
        scf.cf.log.add_config(c)
        c.start()
    return confs


def set_param_async(cf, full_name, value):
    """Set a parameter; returns (event set when the drone confirms, cleanup)."""
    group, name = full_name.split('.')
    ev = threading.Event()

    def cb(n, v):
        try:
            if int(float(v)) == int(value):
                ev.set()
        except ValueError:
            pass

    cf.param.add_update_callback(group=group, name=name, cb=cb)
    cf.param.set_value(full_name, str(value))

    def cleanup():
        try:
            cf.param.remove_update_callback(group=group, name=name, cb=cb)
        except Exception:
            pass

    return ev, cleanup


def set_param_sync(cf, full_name, value, timeout=1.0):
    ev, cleanup = set_param_async(cf, full_name, value)
    ok = ev.wait(timeout)
    cleanup()
    return ok


def wait_for(pred, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


def upload_trajectory(cf, trajectory_id, trajectory):
    traj_mem = cf.mem.get_mems(MemoryElement.TYPE_TRAJ)[0]
    traj_mem.trajectory = []
    total = 0.0
    for row in trajectory:
        x = Poly4D.Poly(row[1:9])
        y = Poly4D.Poly(row[9:17])
        z = Poly4D.Poly(row[17:25])
        yaw = Poly4D.Poly(row[25:33])
        traj_mem.trajectory.append(Poly4D(row[0], x, y, z, yaw))
        total += row[0]
    if not traj_mem.write_data_sync():
        return None
    cf.high_level_commander.define_trajectory(trajectory_id, 0, len(traj_mem.trajectory))
    return total


def send(cf, tele, axis, rate, thrust):
    """One raw setpoint: rate on the flip axis, the other axis held at 0."""
    roll_rate, pitch_rate = (rate, 0) if axis == 'roll' else (0, rate)
    with tele.lock:
        tele.sent = (roll_rate, pitch_rate, thrust)
    cf.commander.send_setpoint(roll_rate, pitch_rate, 0, int(thrust))


def measure_hover_thrust(tele, duration=1.0):
    samples = []
    end = time.time() + duration
    while time.time() < end:
        with tele.lock:
            samples.append(tele.thrust)
        time.sleep(0.02)
    samples = [s for s in samples if s > 1000]
    return sum(samples) / len(samples) if samples else None


def flip(cf, tele, hover, axis, direction, label):
    """One full 360 deg flip on 'pitch' (backflip) or 'roll' (side flip)."""
    up = min(hover * UP_FACTOR, MAX_THRUST)
    inv = min(hover * INVERTED_FACTOR, MAX_THRUST)
    boost = min(hover * BOOST_FACTOR, MAX_THRUST)
    param = AXES[axis]['param']

    if not set_param_sync(cf, param, 0):
        log.error('%s: drone did not confirm rate mode. Skipping it.', label)
        return False

    tele.start_tracking()
    start_z = tele.snapshot(axis)[0]

    # 1. Boost until rising fast enough.
    tele.phase = f'{label}_boost'
    t0 = time.time()
    while True:
        z, vz, _, _, _ = tele.snapshot(axis)
        if vz >= ENTRY_VZ or z >= MAX_Z - 0.4 or time.time() - t0 > BOOST_MAX_TIME:
            break
        send(cf, tele, axis, 0, boost)
        time.sleep(DT)
    entry_z, entry_vz = tele.snapshot(axis)[:2]

    # 2. Rotate. Progress uses the magnitude of the measured rotation, so it
    #    works whichever way the axis sign happens to point.
    tele.phase = f'{label}_rotate'
    tele.start_rotation(axis)
    t1 = time.time()
    timed_out = False
    while True:
        _, _, _, rate_now, rot = tele.snapshot(axis)
        rot, rate_now = abs(rot), abs(rate_now)
        if time.time() - t1 > MAX_FLIP_TIME:
            timed_out = True
            break
        if rot >= HANDOVER_MIN and (rate_now <= HANDOVER_GYRO or rot >= HANDOVER_MAX):
            break
        thrust = up if (rot < EARLY_UP_END or rot >= UP_AGAIN_ANGLE) else inv
        if rot < BRAKE_ANGLE:
            rate = FLIP_RATE
        elif rot < HANDOVER_MIN:
            rate = BRAKE_RATE
        else:
            rate = 0
        send(cf, tele, axis, direction * rate, thrust)
        time.sleep(DT)
    rot_time = time.time() - t1
    handover_rot = abs(tele.snapshot(axis)[4])

    # 3. Back to angle mode, full thrust until level and rising.
    tele.phase = f'{label}_recover'
    ev, cleanup = set_param_async(cf, param, 1)
    t2 = time.time()
    while time.time() - t2 < MAX_RECOVER_TIME:
        _, vz, angle, _, _ = tele.snapshot(axis)
        elapsed = time.time() - t2
        if ev.is_set() and elapsed > MIN_RECOVER_TIME and vz > 0 and abs(angle) < 15:
            break
        send(cf, tele, axis, 0, MAX_THRUST)
        time.sleep(DT)
    rec_time = time.time() - t2
    confirmed = ev.is_set()
    cleanup()
    tele.stop_rotation()
    tele.stop_tracking()

    with tele.lock:
        apex, low, peak = tele.max_z, tele.min_z, abs(tele.rot_max)
    log.info('[%s] entry vz=%.2f m/s at z=%.2f m', label, entry_vz, entry_z)
    log.info('[%s] handed over at %.0f deg after %.2f s%s, peak %.0f deg (overshoot %.0f)',
             label, handover_rot, rot_time, ' (TIMED OUT)' if timed_out else '',
             peak, max(0, peak - 360))
    log.info('[%s] recovery %.2f s; apex %.2f m, lowest %.2f m, lost %.2f m (estimator)',
             label, rec_time, apex, low, start_z - low)
    if not confirmed:
        log.warning('[%s] angle-mode switch was not confirmed in time.', label)
    return True


def catch_and_settle(cf, tele, hl, sx, sy, target_z, label):
    """Hold position right after a flip, hand back to the HL commander,
    hover, and report whether it's steady enough for another flip."""
    tele.phase = f'{label}_catch'
    for _ in range(int(CATCH_TIME * RATE_HZ)):
        cf.commander.send_position_setpoint(sx, sy, target_z, 0.0)
        time.sleep(DT)
    cf.commander.send_notify_setpoint_stop()
    time.sleep(0.1)

    tele.phase = f'{label}_settle'
    hl.go_to(sx, sy, target_z, 0.0, 1.5)
    time.sleep(SETTLE_TIME)
    with tele.lock:
        x, y, z, vz = tele.x, tele.y, tele.z, tele.vz
    err = ((x - sx) ** 2 + (y - sy) ** 2 + (z - target_z) ** 2) ** 0.5
    ok = err <= SETTLE_MAX_POS_ERR and abs(vz) <= SETTLE_MAX_VZ
    log.info('[%s] settled: %.2f m from the hover point, vz=%.2f m/s -> %s',
             label, err, vz, 'OK' if ok else 'NOT steady')
    return ok


def get_hover_thrust(tele):
    hover = measure_hover_thrust(tele)
    if hover is None:
        log.warning('Could not measure hover thrust, using %s.', FALLBACK_HOVER_THRUST)
        return FALLBACK_HOVER_THRUST
    log.info('Hover thrust: %.0f (battery %.2f V under load)', hover, tele.vbat or 0)
    return hover


def flip_allowed(tele, hover, label):
    with tele.lock:
        vbat, z = tele.vbat, tele.z
    if vbat is not None and vbat < MIN_BATTERY_LOADED_V:
        log.warning('Skipping %s: battery %.2f V < %.2f V under load.',
                    label, vbat, MIN_BATTERY_LOADED_V)
        return False
    if hover > MAX_THRUST / 1.3:
        log.warning('Skipping %s: hover thrust %.0f leaves too little power to recover.',
                    label, hover)
        return False
    if z > MAX_Z + 0.2:
        log.warning('Skipping %s: too high (z=%.2f).', label, z)
        return False
    return True


def main(do_backflip, do_side_flip):
    cflib.crtp.init_drivers()
    log.info('Connecting to %s ...', URI)
    scf = SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache'))
    try:
        scf.open_link()
    except Exception as e:
        log.error('Could not connect: %s', e)
        log.error('Close cfclient (only one program can use the Crazyradio) and retry.')
        sys.exit(1)

    cf = scf.cf
    hl = cf.high_level_commander
    tele = Telemetry()
    log_confs = []
    armed = False
    sz = None
    try:
        log.info('Uploading figure-8 trajectory...')
        fig8_duration = upload_trajectory(cf, TRAJ_ID, FIGURE8)
        if fig8_duration is None:
            log.error('Trajectory upload failed. Aborting.')
            return
        fig8_time = fig8_duration * FIG8_TIME_SCALE
        log.info('Figure 8 takes %.1f s', fig8_time)

        for p in (AXES['pitch']['param'], AXES['roll']['param']):
            set_param_sync(cf, p, 1)

        log.info('Waiting for the position estimate to settle...')
        reset_estimator(scf)
        log_confs = start_telemetry(scf, tele)
        if not wait_for(lambda: tele.received and tele.vbat is not None
                        and tele.sup is not None, 2.0):
            log.error('No telemetry received. Aborting.')
            return

        vbat = tele.vbat
        log.info('Battery: %.2f V (min %.2f V)', vbat, MIN_BATTERY_V)
        if vbat < MIN_BATTERY_V:
            log.error('Battery too low. Charge it first.')
            return

        if not (tele.sup & 0b11):        # bit 0 = can be armed, bit 1 = armed
            log.error('Drone reports it cannot be armed (supervisor.info=%d). '
                      'Check the boot console (e.g. Lighthouse deck not booting) '
                      'and run check_supervisor.py.', tele.sup)
            return

        with tele.lock:
            sx, sy, sz = tele.x, tele.y, tele.z
        target_z = min(sz + CLIMB_ABOVE_START, MAX_Z)
        log.info('Start (%.2f, %.2f, %.2f) -> hover at z=%.2f', sx, sy, sz, target_z)
        if target_z - sz < 0.5:
            log.error('Not enough room below MAX_Z. Lower the start point or raise MAX_Z.')
            return

        cf.supervisor.send_arming_request(True)
        if not wait_for(lambda: tele.sup is not None and tele.sup & 0b10, 1.5):
            log.error('Arming was refused (supervisor.info=%s). Aborting.', tele.sup)
            return
        armed = True
        cf.commander.send_setpoint(0, 0, 0, 0)       # unlock raw setpoints
        cf.commander.send_notify_setpoint_stop()
        time.sleep(0.2)

        tele.phase = 'takeoff'
        log.info('Taking off...')
        hl.takeoff(target_z, TAKEOFF_TIME)
        time.sleep(TAKEOFF_TIME + 0.5)
        hl.go_to(sx, sy, target_z, 0.0, 1.0)
        time.sleep(1.5)

        # ---- Figure 8 ----
        tele.phase = 'figure8'
        log.info('Flying the figure 8...')
        hl.start_trajectory(TRAJ_ID, FIG8_TIME_SCALE, relative_position=True)
        time.sleep(fig8_time + 0.5)
        tele.phase = 'return'
        hl.go_to(sx, sy, target_z, 0.0, 2.0)
        time.sleep(2.5)

        steady = True

        # ---- Backflip ----
        if do_backflip:
            tele.phase = 'hover'
            hover = get_hover_thrust(tele)
            if flip_allowed(tele, hover, 'backflip'):
                log.info('Backflip!')
                flip(cf, tele, hover, 'pitch', BACKFLIP_DIRECTION, 'backflip')
                steady = catch_and_settle(cf, tele, hl, sx, sy, target_z, 'backflip')

        # ---- Side flip (optional) ----
        if do_side_flip:
            if not steady:
                log.warning('Skipping the side flip: not steady after the backflip.')
            else:
                tele.phase = 'hover'
                hover = get_hover_thrust(tele)
                if flip_allowed(tele, hover, 'side flip'):
                    log.info('Side flip!')
                    flip(cf, tele, hover, 'roll', SIDE_FLIP_DIRECTION, 'sideflip')
                    catch_and_settle(cf, tele, hl, sx, sy, target_z, 'sideflip')

        # ---- Land ----
        tele.phase = 'land'
        log.info('Landing...')
        hl.land(sz, 3.0)
        time.sleep(3.5)
        hl.stop()
    except KeyboardInterrupt:
        log.warning('Interrupted: landing. Press Ctrl+C again to cut the motors.')
        try:
            if armed and sz is not None:
                for p in (AXES['pitch']['param'], AXES['roll']['param']):
                    cf.param.set_value(p, '1')
                cf.commander.send_notify_setpoint_stop()
                tele.phase = 'abort_land'
                hl.land(sz, 2.5)
                time.sleep(3.0)
                hl.stop()
        except KeyboardInterrupt:
            log.warning('Cutting the motors.')
    finally:
        try:
            for p in (AXES['pitch']['param'], AXES['roll']['param']):
                cf.param.set_value(p, '1')
            cf.commander.send_stop_setpoint()
            cf.commander.send_notify_setpoint_stop()
            time.sleep(0.1)
            for lc in log_confs:         # stop streams cleanly on the drone
                try:
                    lc.stop()
                except Exception:
                    pass
            time.sleep(0.1)
        finally:
            tele.save(LOG_FILE)
            scf.close_link()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crazyflie figure 8 + flips')
    parser.add_argument('--no-flips', action='store_true',
                        help='Figure 8 only, then land (test run).')
    parser.add_argument('--no-side-flip', action='store_true',
                        help='Skip the side flip.')
    args = parser.parse_args()
    main(do_backflip=DO_BACKFLIP and not args.no_flips,
         do_side_flip=DO_SIDE_FLIP and not args.no_flips and not args.no_side_flip)