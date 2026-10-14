"""
Crazyflie show: figure 8 -> backflip -> side flip (optional) -> land.  v3

    python show.py                  # figure 8, backflip, side flip, land
    python show.py --no-side-flip   # figure 8, backflip, land
    python show.py --no-flips       # figure 8 only (test this FIRST)

Or set DO_SIDE_FLIP = False below to turn the side flip off permanently.

Flying flips with the Flow deck on
----------------------------------
The Kalman filter keeps fusing the downward z-ranger until ~84 deg of tilt
and there is no parameter to stop it. On its side the sensor sees a wall,
so the estimate (height AND attitude) gets corrupted mid-flip. In
logs/show_20261009_134047.csv the roll estimate ended ~60 deg wrong and
the angle-mode recovery flew the drone into the floor.

So the side flip no longer trusts the estimate until it has been checked:
  1. Both roll and pitch are in RATE mode for the whole flip: nothing
     uses the estimated attitude while the drone is rotating.
  2. "gyro" recovery: after the flip the drone levels out on its own gyro
     (the rotation this script integrates on both axes), not the estimate.
  3. Then the estimate is checked. If it jumped during the flip, or still
     disagrees with "level" by more than ATT_DISAGREE_DEG, the Kalman filter
     is restarted at the last good position with the pre-flip yaw.
     Lighthouse + the z-ranger (now pointing at the floor again) pull it
     back within a fraction of a second.
  4. Only then angle mode, then position hold.
The backflip keeps the "angle" recovery that already works in your logs,
but switches to the gyro path automatically if the estimate jumps.
Flow (optical-flow) measurements are also switched off during each flip.

Space: the figure 8 swings about +-0.95 m in x and +-0.5 m in y around the
takeoff point, and each flip drifts ~0.5 m. Keep ~3 x 2 m clear and stay
inside Lighthouse coverage. Fully charged battery, hand on Ctrl+C.

Ctrl+C once -> tries to land.  Ctrl+C twice -> cuts the motors.
Logs: console via run_log, CSV at logs/show_flight_YYYYmmdd_HHMMSS.csv.
"""
import argparse
import math
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

import run_log

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7EA')
log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()
LOG_FILE = run_log.csv_path('flight', script=__file__, when=STARTED_AT)

# ---- What to fly -----------------------------------------------------------
DO_BACKFLIP = True
DO_SIDE_FLIP = True            # <- set False to skip the side flip
BACKFLIP_DIRECTION = 1         # -1 = forward flip instead
SIDE_FLIP_DIRECTION = 1        # -1 = roll the other way
BACKFLIP_RECOVERY = 'angle'    # 'angle' (proven) or 'gyro'
SIDE_FLIP_RECOVERY = 'gyro'    # 'gyro' (safe with the Flow deck) or 'angle'

# ---- Height ----------------------------------------------------------------
CLIMB_ABOVE_START = 1.0        # m above the takeoff point
MAX_Z = 2.0                    # m, hard cap: keep below your ceiling
TAKEOFF_TIME = 2.5             # s

# ---- Figure 8 --------------------------------------------------------------
FIG8_TIME_SCALE = 1.0          # >1 = slower (1.5 = 50 % slower)
TRAJ_ID = 1

# ---- Flip tuning (same as flip.py v5) --------------------------------------
ENTRY_VZ = 1.0                 # m/s upward speed before rotating
BOOST_FACTOR = 1.7             # x hover thrust while boosting
BOOST_MAX_TIME = 0.45          # s
FLIP_RATE = 1000               # deg/s commanded rotation rate
MAX_FLIP_TIME = 1.0            # s, safety limit for the rotation
UP_FACTOR = 1.5                # x hover thrust when roughly upright
INVERTED_FACTOR = 0.45         # x hover thrust when inverted
EARLY_UP_END = 50              # deg: high thrust until here...
UP_AGAIN_ANGLE = 245           # deg: ...and again from here
BRAKE_ANGLE = 235              # deg: start slowing the rotation
BRAKE_RATE = 250               # deg/s while braking
HANDOVER_MIN = 290             # deg: earliest hand-over
HANDOVER_GYRO = 350            # deg/s: hand over once slower than this...
HANDOVER_MAX = 345             # deg: ...or at this angle regardless
MAX_THRUST = 60000
MIN_RECOVER_TIME = 0.15        # s at full power, at least ('angle' recovery)
MAX_RECOVER_TIME = 0.8         # s at full power, at most ('angle' recovery)

# ---- Gyro recovery ---------------------------------------------------------
LEVEL_KP = 8.0                 # deg/s commanded per deg of tilt left
LEVEL_MAX_RATE = 400           # deg/s
LEVEL_TOL_DEG = 5              # level when both axes are within this...
LEVEL_RATE_TOL = 60            # ...and both rates are below this (deg/s)
LEVEL_MAX_TIME = 0.5           # s, then hand back regardless
ATT_DISAGREE_DEG = 20          # restart the estimator if it is further off level
HANDBACK_HOLD = 0.35           # s level at ~hover thrust after the hand-back

# ---- Estimator protection --------------------------------------------------
DISABLE_FLOW_DURING_FLIPS = True
# An estimator jump is any 10 ms sample beyond these. The good backflip in
# the log peaked at |vz| 2.2 m/s, 0.20 m z step, 0.14 m sideways step; the
# crashed side flip hit 6.1 m/s, 0.77 m and 0.84 m.
GLITCH_VZ = 3.5                # m/s
GLITCH_DZ = 0.30               # m per sample
GLITCH_DXY = 0.35              # m per sample

# ---- Between flips ---------------------------------------------------------
CATCH_TIME = 1.5               # s of position setpoints right after a flip
SETTLE_TIME = 3.0              # s hovering before the next flip
SETTLE_WINDOW = 1.0            # s at the end of SETTLE_TIME that must be steady
SETTLE_MAX_POS_ERR = 0.35      # m from the hover point, worst case in the window
SETTLE_MAX_VZ = 0.25           # m/s, worst case in the window

# ---- Battery ---------------------------------------------------------------
MIN_BATTERY_V = 3.5            # on the ground, before takeoff
MIN_BATTERY_LOADED_V = 3.3     # in the air (voltage sags under load)
FALLBACK_HOVER_THRUST = 35000
RATE_HZ = 100
# ----------------------------------------------------------------------------

DT = 1.0 / RATE_HZ
STAB_PARAMS = ('flightmode.stabModeRoll', 'flightmode.stabModePitch')

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

CSV_FIELDS = ['drone_ms', 'phase', 'x', 'y', 'z', 'vz', 'pitch_deg', 'roll_deg',
              'yaw_deg', 'gyro_x_dps', 'gyro_y_dps', 'rot_roll_deg', 'rot_pitch_deg',
              'cmd_thrust', 'vbat', 'glitch', 'sent_roll', 'sent_pitch', 'sent_thrust']


def wrap360(a):
    """Signed distance from the nearest whole number of turns (deg)."""
    return a - 360.0 * round(a / 360.0)


class Telemetry:
    """State + gyro at 10 ms, thrust/battery/supervisor at 20 ms, the gyro
    integration used during flips, and the estimator-jump watchdog."""

    def __init__(self):
        self.lock = threading.Lock()
        self.x = self.y = self.z = self.vz = 0.0
        self.pitch = self.roll = self.yaw = 0.0
        self.gx = self.gy = 0.0
        self.thrust = 0.0
        self.vbat = None
        self.sup = None
        self.sent = (0.0, 0.0, 0.0)
        self.last_gyro_ts = None
        self.rot_roll = self.rot_pitch = 0.0   # integrated gyro, deg
        self.flip_axis = 'pitch'
        self.rot_max = 0.0
        self.integrating = False
        self.tracking = False
        self.min_z = self.max_z = 0.0
        self.watching = False
        self.prev_pos = None
        self.glitch = None
        self.phase = 'ground'
        self.rows = []
        self.received = False

    def state_cb(self, ts, d, lc):
        with self.lock:
            x, y = d['stateEstimate.x'], d['stateEstimate.y']
            z, vz = d['stateEstimate.z'], d['stateEstimate.vz']
            if self.watching and self.glitch is None and self.prev_pos is not None:
                px, py, pz = self.prev_pos
                dz, dxy = abs(z - pz), math.hypot(x - px, y - py)
                if abs(vz) > GLITCH_VZ or dz > GLITCH_DZ or dxy > GLITCH_DXY:
                    self.glitch = {'ms': ts, 'phase': self.phase, 'vz': vz, 'dz': dz,
                                   'dxy': dxy, 'rotation': self._flip_rot(),
                                   'last_good': (px, py, pz)}
            self.prev_pos = (x, y, z)
            self.x, self.y, self.z, self.vz = x, y, z, vz
            self.pitch, self.roll = d['stabilizer.pitch'], d['stabilizer.roll']
            if self.tracking:
                self.min_z = min(self.min_z, z)
                self.max_z = max(self.max_z, z)
            self.received = True
            self.rows.append([ts, self.phase, x, y, z, vz, self.pitch, self.roll,
                              self.yaw, self.gx, self.gy, self.rot_roll, self.rot_pitch,
                              self.thrust, self.vbat,
                              int(self.watching and self.glitch is not None), *self.sent])

    def gyro_cb(self, ts, d, lc):
        with self.lock:
            self.gx, self.gy = d['gyro.x'], d['gyro.y']
            self.yaw = d['stabilizer.yaw']
            if self.integrating and self.last_gyro_ts is not None:
                dt = (ts - self.last_gyro_ts) / 1000.0
                self.rot_roll += self.gx * dt
                self.rot_pitch += self.gy * dt
                if abs(self._flip_rot()) > abs(self.rot_max):
                    self.rot_max = self._flip_rot()
            self.last_gyro_ts = ts

    def slow_cb(self, ts, d, lc):
        with self.lock:
            self.thrust = d['controller.cmd_thrust']
            self.vbat = d['pm.vbat']
            self.sup = int(d['supervisor.info'])

    def _flip_rot(self):
        return self.rot_roll if self.flip_axis == 'roll' else self.rot_pitch

    def start_flip(self, axis):
        """Begin z tracking, gyro integration and the estimator watchdog."""
        with self.lock:
            self.flip_axis = axis
            self.rot_roll = self.rot_pitch = self.rot_max = 0.0
            self.integrating = True
            self.min_z = self.max_z = self.z
            self.tracking = True
            self.glitch = None
            self.prev_pos = (self.x, self.y, self.z)
            self.watching = True

    def end_flip(self):
        with self.lock:
            self.tracking = self.watching = self.integrating = False

    def get_glitch(self):
        with self.lock:
            return self.glitch

    def flip_state(self):
        """z, vz, flip-axis gyro rate, flip-axis integrated rotation."""
        with self.lock:
            rate = self.gx if self.flip_axis == 'roll' else self.gy
            return self.z, self.vz, rate, self._flip_rot()

    def gyro_state(self):
        """Integrated roll, pitch (deg) and their rates (deg/s)."""
        with self.lock:
            return self.rot_roll, self.rot_pitch, self.gx, self.gy

    def attitude(self):
        with self.lock:
            return self.roll, self.pitch, self.yaw

    def position(self):
        with self.lock:
            return self.x, self.y, self.z, self.vz

    def save(self, path):
        with self.lock:
            rows = list(self.rows)
        written = run_log.write_csv(path, CSV_FIELDS, rows, script=__file__,
                                    generated_at=datetime.now().astimezone())
        log.info('Flight log written to %s (%d samples)', written, len(rows))


def start_telemetry(scf, tele):
    state = LogConfig(name='ShowState', period_in_ms=10)
    for name in ('stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
                 'stateEstimate.vz', 'stabilizer.pitch', 'stabilizer.roll'):
        state.add_variable(name, 'float')
    state.data_received_cb.add_callback(tele.state_cb)

    gyro = LogConfig(name='ShowGyro', period_in_ms=10)
    for name in ('gyro.x', 'gyro.y', 'stabilizer.yaw'):
        gyro.add_variable(name, 'float')
    gyro.data_received_cb.add_callback(tele.gyro_cb)

    slow = LogConfig(name='ShowSlow', period_in_ms=20)
    slow.add_variable('controller.cmd_thrust', 'float')
    slow.add_variable('pm.vbat', 'float')
    slow.add_variable('supervisor.info', 'uint16_t')
    slow.data_received_cb.add_callback(tele.slow_cb)

    confs = [state, gyro, slow]
    for c in confs:
        scf.cf.log.add_config(c)
        c.start()
    return confs


# ---- Small helpers ---------------------------------------------------------

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


def set_stab_mode(cf, mode, timeout=1.0):
    """Roll and pitch together: 0 = rate mode, 1 = angle mode."""
    pending = [set_param_async(cf, p, mode) for p in STAB_PARAMS]
    end = time.time() + timeout
    ok = all(ev.wait(max(0.0, end - time.time())) for ev, _ in pending)
    for _, cleanup in pending:
        cleanup()
    return ok


def wait_for(pred, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


def fitted_decks(cf):
    """Names of downward-sensing decks that are fitted (Flow v1/v2, z-ranger)."""
    found = []
    for name in ('deck.bcFlow', 'deck.bcFlow2', 'deck.bcZRanger', 'deck.bcZRanger2'):
        try:
            if int(float(cf.param.get_value(name))):
                found.append(name.split('.')[1])
        except Exception:
            pass                       # parameter absent = deck not fitted
    return found


def set_flow(cf, enabled):
    """Turn the Flow deck's optical-flow measurements on or off."""
    if not set_param_sync(cf, 'motion.disable', 0 if enabled else 1, timeout=0.5):
        log.warning('Flow deck did not confirm motion.disable=%d.', 0 if enabled else 1)


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


def send(cf, tele, roll, pitch, thrust):
    """One raw setpoint. Rate mode: deg/s. Angle mode: deg (0 = level)."""
    with tele.lock:
        tele.sent = (roll, pitch, thrust)
    cf.commander.send_setpoint(roll, pitch, 0, int(thrust))


def send_axis(cf, tele, axis, value, thrust):
    """Setpoint on the flip axis only; the other axis held at 0."""
    if axis == 'roll':
        send(cf, tele, value, 0, thrust)
    else:
        send(cf, tele, 0, value, thrust)


def measure_hover_thrust(tele, duration=1.0):
    samples = []
    end = time.time() + duration
    while time.time() < end:
        with tele.lock:
            samples.append(tele.thrust)
        time.sleep(0.02)
    samples = [s for s in samples if s > 1000]
    if not samples:
        log.warning('Could not measure hover thrust, using %s.', FALLBACK_HOVER_THRUST)
        return FALLBACK_HOVER_THRUST
    hover = sum(samples) / len(samples)
    log.info('Hover thrust %.0f (battery %.2f V under load)', hover, tele.vbat or 0)
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


# ---- The flip --------------------------------------------------------------

def flip(cf, tele, hover, axis, direction, label, recovery):
    """One 360 deg flip on 'pitch' (backflip) or 'roll' (side flip).

    recovery: 'angle' = angle mode + full power until level (estimator-based),
              'gyro'  = level out on the gyro, check / restart the estimator,
                        then angle mode. Used automatically if the estimate
                        jumps.
    Returns a result dict, or None if the flip could not start.
    """
    up = min(hover * UP_FACTOR, MAX_THRUST)
    inv = min(hover * INVERTED_FACTOR, MAX_THRUST)
    boost = min(hover * BOOST_FACTOR, MAX_THRUST)

    # Rate mode on BOTH axes: nothing uses the estimated attitude mid-flip.
    if not set_stab_mode(cf, 0):
        log.error('[%s] drone did not confirm rate mode. Skipping it.', label)
        set_stab_mode(cf, 1)
        return None

    x0, y0, z0, _ = tele.position()
    yaw_before = tele.attitude()[2]
    tele.start_flip(axis)

    # 1. Boost until rising fast enough (rate 0 = hold attitude on the gyro).
    tele.phase = f'{label}_boost'
    t0 = time.time()
    while True:
        z, vz, _, _ = tele.flip_state()
        if vz >= ENTRY_VZ or z >= MAX_Z - 0.4 or time.time() - t0 > BOOST_MAX_TIME:
            break
        send(cf, tele, 0, 0, boost)
        time.sleep(DT)
    entry_z, entry_vz = tele.flip_state()[:2]

    # 2. Rotate on the gyro, braking before the hand-over.
    tele.phase = f'{label}_rotate'
    t1 = time.time()
    timed_out = False
    while True:
        _, _, rate_now, rot = tele.flip_state()
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
        send_axis(cf, tele, axis, direction * rate, thrust)
        time.sleep(DT)
    rot_time = time.time() - t1
    handover_rot = abs(tele.flip_state()[3])

    # 3. Recover.
    if tele.get_glitch() and recovery != 'gyro':
        log.warning('[%s] estimate jumped during the flip: switching to gyro recovery.',
                    label)
        recovery = 'gyro'
    if recovery == 'angle':
        if not angle_recovery(cf, tele, axis, label):
            log.warning('[%s] estimate jumped during recovery: switching to gyro recovery.',
                        label)
            set_stab_mode(cf, 0, timeout=0.3)
            recovery = 'gyro'
    restarted = False
    if recovery == 'gyro':
        level_on_gyro(cf, tele, up, label)
        restarted = hand_back(cf, tele, hover, label, (x0, y0, z0), yaw_before)

    with tele.lock:
        apex, low, peak = tele.max_z, tele.min_z, abs(tele.rot_max)
    glitch = tele.get_glitch()
    tele.end_flip()

    log.info('[%s] entry vz=%.2f m/s at z=%.2f m', label, entry_vz, entry_z)
    log.info('[%s] handed over at %.0f deg after %.2f s%s, peak %.0f deg (overshoot %.0f)',
             label, handover_rot, rot_time, ' (TIMED OUT)' if timed_out else '',
             peak, max(0, peak - 360))
    if glitch:
        log.info('[%s] estimate jumped at %.0f deg (%s): vz=%.1f m/s, z step %.2f m, '
                 'side step %.2f m. Altitude figures for this flip are not valid.',
                 label, abs(glitch['rotation']), glitch['phase'], glitch['vz'],
                 glitch['dz'], glitch['dxy'])
    else:
        log.info('[%s] apex %.2f m, lowest %.2f m, lost %.2f m (estimator)',
                 label, apex, low, z0 - low)
    return {'label': label, 'recovery': recovery, 'glitch': glitch is not None,
            'restarted': restarted, 'timed_out': timed_out, 'peak': peak}


def angle_recovery(cf, tele, axis, label):
    """Angle mode + full thrust until level and rising. Returns False if the
    estimate jumped meanwhile (the caller then switches to gyro recovery)."""
    tele.phase = f'{label}_recover'
    pending = [set_param_async(cf, p, 1) for p in STAB_PARAMS]
    t0 = time.time()
    ok = True
    while time.time() - t0 < MAX_RECOVER_TIME:
        if tele.get_glitch():
            ok = False
            break
        _, vz, _, _ = tele.flip_state()
        roll, pitch, _ = tele.attitude()
        angle = roll if axis == 'roll' else pitch
        confirmed = all(ev.is_set() for ev, _ in pending)
        if (confirmed and time.time() - t0 > MIN_RECOVER_TIME
                and vz > 0 and abs(angle) < 15):
            break
        send(cf, tele, 0, 0, MAX_THRUST)
        time.sleep(DT)
    if not all(ev.is_set() for ev, _ in pending):
        log.warning('[%s] angle-mode switch was not confirmed in time.', label)
    for _, cleanup in pending:
        cleanup()
    return ok


def level_on_gyro(cf, tele, thrust, label):
    """Rate mode: steer both axes back to a whole number of turns using the
    integrated gyro, i.e. level, without looking at the estimate."""
    tele.phase = f'{label}_level'
    t0 = time.time()
    while True:
        rr, rp, gx, gy = tele.gyro_state()
        er, ep = wrap360(rr), wrap360(rp)
        done = (abs(er) < LEVEL_TOL_DEG and abs(ep) < LEVEL_TOL_DEG
                and abs(gx) < LEVEL_RATE_TOL and abs(gy) < LEVEL_RATE_TOL)
        if done or time.time() - t0 > LEVEL_MAX_TIME:
            break
        cmd_r = max(-LEVEL_MAX_RATE, min(LEVEL_MAX_RATE, -LEVEL_KP * er))
        cmd_p = max(-LEVEL_MAX_RATE, min(LEVEL_MAX_RATE, -LEVEL_KP * ep))
        send(cf, tele, cmd_r, cmd_p, thrust)
        time.sleep(DT)
    log.info('[%s] gyro levelling %s in %.2f s (roll %.0f deg, pitch %.0f deg off)',
             label, 'done' if done else 'TIMED OUT', time.time() - t0, er, ep)


def hand_back(cf, tele, hover, label, pre_flip_pos, yaw_before):
    """Check the estimate against the gyro (which says: level). Restart the
    Kalman filter if it jumped or disagrees, then switch to angle mode and
    hold level briefly. Returns True if the estimator was restarted."""
    roll, pitch, _ = tele.attitude()
    glitch = tele.get_glitch()
    off = max(abs(wrap360(roll)), abs(wrap360(pitch)))
    restart = glitch is not None or off > ATT_DISAGREE_DEG
    if restart:
        if glitch:
            gx, gy, gz = glitch['last_good']
            reason = 'it jumped during the flip'
        else:
            gx, gy, gz = tele.position()[:3]
            reason = f'it is {off:.0f} deg off level'
        # z: the last trusted height, never above the pre-flip height
        gz = min(gz, pre_flip_pos[2])
        log.warning('[%s] restarting the estimator (%s) at (%.2f, %.2f, %.2f), yaw %.0f deg.',
                    label, reason, gx, gy, gz, yaw_before)
        cf.param.set_value('kalman.initialX', str(gx))
        cf.param.set_value('kalman.initialY', str(gy))
        cf.param.set_value('kalman.initialZ', str(gz))
        cf.param.set_value('kalman.initialYaw', str(math.radians(yaw_before)))
        cf.param.set_value('kalman.resetEstimation', '1')
    else:
        log.info('[%s] estimate agrees with the gyro (%.0f deg off level): keeping it.',
                 label, off)

    tele.phase = f'{label}_handback'
    pending = [set_param_async(cf, p, 1) for p in STAB_PARAMS]
    t0 = time.time()
    cleared = not restart
    while time.time() - t0 < HANDBACK_HOLD:
        if not cleared and time.time() - t0 > 0.1:
            cf.param.set_value('kalman.resetEstimation', '0')
            cleared = True
        send(cf, tele, 0, 0, hover * 1.1)   # rate 0 / level, slightly above hover
        time.sleep(DT)
    if not all(ev.is_set() for ev, _ in pending):
        log.warning('[%s] angle-mode switch was not confirmed in time.', label)
    for _, cleanup in pending:
        cleanup()
    return restart


def catch(cf, tele, home, label):
    """Hold the hover point with position setpoints, then hand back to the
    high-level commander."""
    sx, sy, hz = home
    tele.phase = f'{label}_catch'
    for _ in range(int(CATCH_TIME * RATE_HZ)):
        cf.commander.send_position_setpoint(sx, sy, hz, 0.0)
        time.sleep(DT)
    cf.commander.send_notify_setpoint_stop()
    time.sleep(0.1)


def settle(tele, hl, home, label):
    """Hover at the hover point; True if the last SETTLE_WINDOW s were steady."""
    sx, sy, hz = home
    tele.phase = f'{label}_settle'
    hl.go_to(sx, sy, hz, 0.0, 1.5)
    worst_err = worst_vz = 0.0
    t0 = time.time()
    while time.time() - t0 < SETTLE_TIME:
        if time.time() - t0 >= SETTLE_TIME - SETTLE_WINDOW:
            x, y, z, vz = tele.position()
            worst_err = max(worst_err, math.dist((x, y, z), home))
            worst_vz = max(worst_vz, abs(vz))
        time.sleep(0.02)
    ok = worst_err <= SETTLE_MAX_POS_ERR and worst_vz <= SETTLE_MAX_VZ
    log.info('[%s] settled: worst %.2f m from the hover point, worst |vz| %.2f m/s -> %s',
             label, worst_err, worst_vz, 'OK' if ok else 'NOT steady')
    return ok


# ---- Main ------------------------------------------------------------------

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
    has_flow = False
    sz = None
    log.info('Connected. Flight CSV will be %s', LOG_FILE)
    try:
        try:
            scf.wait_for_params()
        except AttributeError:
            time.sleep(1.0)                 # older cflib
        decks = fitted_decks(cf)
        has_flow = any(d.startswith('bcFlow') for d in decks)
        log.info('Downward decks fitted: %s', ', '.join(decks) or 'none')
        if decks and SIDE_FLIP_RECOVERY != 'gyro' and do_side_flip:
            log.warning("SIDE_FLIP_RECOVERY='angle' with a z-ranger fitted is what crashed "
                        "before. 'gyro' is recommended.")

        log.info('Uploading figure-8 trajectory...')
        fig8_duration = upload_trajectory(cf, TRAJ_ID, FIGURE8)
        if fig8_duration is None:
            log.error('Trajectory upload failed. Aborting.')
            return
        fig8_time = fig8_duration * FIG8_TIME_SCALE
        log.info('Figure 8 takes %.1f s', fig8_time)

        set_stab_mode(cf, 1)
        if has_flow:
            set_flow(cf, True)

        log.info('Waiting for the position estimate to settle...')
        reset_estimator(scf)
        log_confs = start_telemetry(scf, tele)
        if not wait_for(lambda: tele.received and tele.vbat is not None
                        and tele.sup is not None, 2.0):
            log.error('No telemetry received. Aborting.')
            return

        log.info('Battery: %.2f V (min %.2f V)', tele.vbat, MIN_BATTERY_V)
        if tele.vbat < MIN_BATTERY_V:
            log.error('Battery too low. Charge it first.')
            return

        if not (tele.sup & 0b11):        # bit 0 = can be armed, bit 1 = armed
            log.error('Drone reports it cannot be armed (supervisor.info=%d). '
                      'Check the boot console and run check_supervisor.py.', tele.sup)
            return

        sx, sy, sz, _ = tele.position()
        target_z = min(sz + CLIMB_ABOVE_START, MAX_Z)
        home = (sx, sy, target_z)
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

        # ---- Flips ----
        flips = []
        if do_backflip:
            flips.append(('backflip', 'pitch', BACKFLIP_DIRECTION, BACKFLIP_RECOVERY))
        if do_side_flip:
            flips.append(('sideflip', 'roll', SIDE_FLIP_DIRECTION, SIDE_FLIP_RECOVERY))

        results = []
        for i, (label, axis, direction, recovery) in enumerate(flips):
            if i > 0 and not settle(tele, hl, home, flips[i - 1][0]):
                log.warning('Skipping the %s: not steady after the %s.',
                            label, flips[i - 1][0])
                break
            tele.phase = f'{label}_hover'
            hover = measure_hover_thrust(tele)
            if not flip_allowed(tele, hover, label):
                break
            if has_flow and DISABLE_FLOW_DURING_FLIPS:
                set_flow(cf, False)
            log.info('%s! (%s recovery)', label.capitalize(), recovery)
            result = flip(cf, tele, hover, axis, direction, label, recovery)
            if result is None:
                break
            results.append(result)
            catch(cf, tele, home, label)
            if has_flow and DISABLE_FLOW_DURING_FLIPS:
                set_flow(cf, True)

        # ---- Land ----
        tele.phase = 'land'
        log.info('Landing...')
        hl.go_to(sx, sy, target_z, 0.0, 1.0)
        time.sleep(1.2)
        hl.land(sz, 3.0)
        time.sleep(3.5)
        hl.stop()

        for r in results:
            log.info('Summary %-8s: peak %.0f deg, %s recovery%s%s%s', r['label'],
                     r['peak'], r['recovery'],
                     ', estimate jumped' if r['glitch'] else '',
                     ', estimator restarted' if r['restarted'] else '',
                     ', TIMED OUT' if r['timed_out'] else '')
    except KeyboardInterrupt:
        log.warning('Interrupted: landing. Press Ctrl+C again to cut the motors.')
        try:
            if armed and sz is not None:
                for p in STAB_PARAMS:
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
            for p in STAB_PARAMS:
                cf.param.set_value(p, '1')
            if has_flow:
                cf.param.set_value('motion.disable', '0')
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