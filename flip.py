"""
Crazyflie backflip (pitch flip) with Lighthouse positioning.  v5

Sequence: connect -> reset estimator -> arm -> take off to a height RELATIVE
to the start -> measure hover thrust -> boost until rising fast enough ->
flip with thrust scheduled by the MEASURED rotation angle -> brake before
the hand-over so it doesn't overshoot level -> full-power recovery ->
position hold -> land.

Every flight writes a CSV under logs/ (script name + timestamp in the
file and header) and prints entry speed, apex, peak rotation and the
lowest altitude reached.

Run `python flip.py --no-flip` FIRST.
UNTESTED: fully charged battery, clear area, hand on the emergency stop.
Expect the drone to drift ~0.5 m sideways during a flip: keep clear space.
"""
import argparse
import sys
import threading
import time
from datetime import datetime

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.utils.reset_estimator import reset_estimator

import run_log

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()
LOG_FILE = run_log.csv_path('flight', script=__file__, when=STARTED_AT)

# ---- Height ---------------------------------------------------------------
CLIMB_ABOVE_START = 1.0   # m above the takeoff point.
MAX_Z = 2.0               # m in Lighthouse frame. Hard cap: set below your ceiling.
TAKEOFF_TIME = 2.5        # s

# ---- Entry (boost) --------------------------------------------------------
ENTRY_VZ = 1.0            # m/s upward speed to reach before rotating. At 1.3 the
                          # drone reached 1.41 m/s and peaked 0.6 m above hover with
                          # no net loss; lower = smaller climb, but more drop.
BOOST_FACTOR = 1.7        # x hover thrust while boosting.
BOOST_MAX_TIME = 0.45     # s, safety limit.

# ---- Flip -----------------------------------------------------------------
FLIP_RATE = 1000          # deg/s commanded pitch rate.
FLIP_DIRECTION = 1        # -1 if it flips the wrong way.
MAX_FLIP_TIME = 1.0       # s, safety limit for the rotation phase.

# Thrust is scheduled by measured rotation angle (0 = upright, 180 = inverted).
# High when the props point up, low when inverted. The 245 deg switch is
# early on purpose: thrust lags ~30-40 ms, i.e. ~35 deg at flip speed.
UP_FACTOR = 1.5           # x hover thrust when roughly upright.
INVERTED_FACTOR = 0.45    # x hover thrust when inverted.
EARLY_UP_END = 50         # deg: high thrust until here...
UP_AGAIN_ANGLE = 245      # deg: ...and again from here.

# Braking: the log showed the drone coasting 53 deg past level after a
# hand-over at 320 deg. Slow the rotation down before handing over.
BRAKE_ANGLE = 235         # deg: start slowing down.
BRAKE_RATE = 250          # deg/s commanded between BRAKE_ANGLE and HANDOVER_MIN.
HANDOVER_MIN = 290        # deg: never hand over earlier (Euler angles break
                          # down around 270 deg, the vertical point).
HANDOVER_GYRO = 350       # deg/s: hand over once rotation >= HANDOVER_MIN and
                          # the rate is below this...
HANDOVER_MAX = 345        # deg: ...or at this angle regardless.

# ---- Recovery -------------------------------------------------------------
MAX_THRUST = 60000        # firmware limit seen in the log.
MIN_RECOVER_TIME = 0.15   # s at full power, at least.
MAX_RECOVER_TIME = 0.8    # s at full power, at most.

FALLBACK_HOVER_THRUST = 35000
MIN_BATTERY_V = 3.9
RATE_HZ = 100
# ---------------------------------------------------------------------------

DT = 1.0 / RATE_HZ


class Telemetry:
    """Fast log stream (10 ms) for control decisions plus the CSV log."""

    def __init__(self):
        self.lock = threading.Lock()
        self.x = self.y = self.z = self.vz = 0.0
        self.pitch = self.gyro_y = self.thrust = 0.0
        self.sent_rate = 0.0
        self.sent_thrust = 0.0
        self.last_ts = None
        self.rotation = 0.0
        self.rot_max = 0.0
        self.integrating = False
        self.tracking = False
        self.min_z = self.max_z = 0.0
        self.phase = 'ground'
        self.rows = []
        self.received = False

    def fast_cb(self, timestamp, data, logconf):
        with self.lock:
            self.x = data['stateEstimate.x']
            self.y = data['stateEstimate.y']
            self.z = data['stateEstimate.z']
            self.vz = data['stateEstimate.vz']
            self.pitch = data['stabilizer.pitch']
            self.gyro_y = data['gyro.y']
            if self.integrating and self.last_ts is not None:
                self.rotation += self.gyro_y * (timestamp - self.last_ts) / 1000.0
                if abs(self.rotation) > abs(self.rot_max):
                    self.rot_max = self.rotation
            self.last_ts = timestamp
            if self.tracking:
                self.min_z = min(self.min_z, self.z)
                self.max_z = max(self.max_z, self.z)
            self.received = True
            self.rows.append([timestamp, self.phase, self.x, self.y, self.z, self.vz,
                              self.pitch, self.gyro_y, self.rotation, self.thrust,
                              self.sent_rate, self.sent_thrust])

    def slow_cb(self, timestamp, data, logconf):
        with self.lock:
            self.thrust = data['controller.cmd_thrust']

    def start_rotation(self):
        with self.lock:
            self.rotation = 0.0
            self.rot_max = 0.0
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

    def snapshot(self):
        with self.lock:
            return self.z, self.vz, self.pitch, self.gyro_y, self.rotation

    def save(self, path):
        with self.lock:
            rows = list(self.rows)
        written = run_log.write_csv(
            path,
            ['drone_ms', 'phase', 'x', 'y', 'z', 'vz', 'pitch_deg',
             'gyro_y_dps', 'rotation_deg', 'cmd_thrust',
             'sent_pitch_rate', 'sent_thrust'],
            rows,
            script=__file__,
            generated_at=datetime.now().astimezone(),
        )
        log.info('Flight log written to %s (%s samples)', written, len(rows))


def start_telemetry(scf, tele):
    fast = LogConfig(name='FlipFast', period_in_ms=10)
    for name in ('stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
                 'stateEstimate.vz', 'stabilizer.pitch', 'gyro.y'):
        fast.add_variable(name, 'float')
    scf.cf.log.add_config(fast)
    fast.data_received_cb.add_callback(tele.fast_cb)
    fast.start()

    slow = LogConfig(name='FlipSlow', period_in_ms=20)
    slow.add_variable('controller.cmd_thrust', 'float')
    scf.cf.log.add_config(slow)
    slow.data_received_cb.add_callback(tele.slow_cb)
    slow.start()
    return [fast, slow]


def read_battery(scf, timeout=3.0):
    result = {}

    def cb(timestamp, data, logconf):
        result['v'] = data['pm.vbat']

    lg = LogConfig(name='Battery', period_in_ms=100)
    lg.add_variable('pm.vbat', 'float')
    scf.cf.log.add_config(lg)
    lg.data_received_cb.add_callback(cb)
    lg.start()
    end = time.time() + timeout
    while 'v' not in result and time.time() < end:
        time.sleep(0.05)
    lg.stop()
    return result.get('v')


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


def send(cf, tele, pitch_rate, thrust):
    """One raw setpoint (roll 0, yaw rate 0); remembered for the CSV log."""
    tele.sent_rate = pitch_rate
    tele.sent_thrust = thrust
    cf.commander.send_setpoint(0, pitch_rate, 0, int(thrust))


def measure_hover_thrust(tele, duration=1.0):
    samples = []
    end = time.time() + duration
    while time.time() < end:
        with tele.lock:
            samples.append(tele.thrust)
        time.sleep(0.02)
    samples = [s for s in samples if s > 1000]
    return sum(samples) / len(samples) if samples else None


def backflip(cf, tele, hover):
    up = min(hover * UP_FACTOR, MAX_THRUST)
    inv = min(hover * INVERTED_FACTOR, MAX_THRUST)
    boost = min(hover * BOOST_FACTOR, MAX_THRUST)

    # Rate mode for pitch (affects raw setpoints only; HL keeps hovering).
    if not set_param_sync(cf, 'flightmode.stabModePitch', 0):
        log.error('Drone did not confirm rate mode. Skipping the flip.')
        return

    tele.start_tracking()
    start_z = tele.snapshot()[0]

    # 1. Boost until it is rising fast enough (battery-independent).
    tele.phase = 'boost'
    t0 = time.time()
    while True:
        z, vz, _, _, _ = tele.snapshot()
        if vz >= ENTRY_VZ or z >= MAX_Z - 0.4 or time.time() - t0 > BOOST_MAX_TIME:
            break
        send(cf, tele, 0, boost)
        time.sleep(DT)
    entry_z, entry_vz = tele.snapshot()[0], tele.snapshot()[1]

    # 2. Rotate. Thrust scheduled by measured angle, braking before hand-over.
    tele.phase = 'rotate'
    tele.start_rotation()
    t1 = time.time()
    timed_out = False
    while True:
        _, _, _, gyro, rot = tele.snapshot()
        rot *= FLIP_DIRECTION
        gyro *= FLIP_DIRECTION
        if time.time() - t1 > MAX_FLIP_TIME:
            timed_out = True
            break
        if rot >= HANDOVER_MIN and (abs(gyro) <= HANDOVER_GYRO or rot >= HANDOVER_MAX):
            break

        thrust = up if (rot < EARLY_UP_END or rot >= UP_AGAIN_ANGLE) else inv
        if rot < BRAKE_ANGLE:
            rate = FLIP_RATE
        elif rot < HANDOVER_MIN:
            rate = BRAKE_RATE
        else:
            rate = 0
        send(cf, tele, FLIP_DIRECTION * rate, thrust)
        time.sleep(DT)
    rot_time = time.time() - t1
    handover_rot = abs(tele.snapshot()[4])

    # 3. Hand over to angle mode and apply FULL thrust. Pitch 0 is safe in
    #    both modes (rate mode: stop rotating; angle mode: level out).
    tele.phase = 'recover'
    ev, cleanup = set_param_async(cf, 'flightmode.stabModePitch', 1)
    t2 = time.time()
    while time.time() - t2 < MAX_RECOVER_TIME:
        _, vz, pitch, _, _ = tele.snapshot()
        elapsed = time.time() - t2
        if ev.is_set() and elapsed > MIN_RECOVER_TIME and vz > 0 and abs(pitch) < 15:
            break
        send(cf, tele, 0, MAX_THRUST)
        time.sleep(DT)
    rec_time = time.time() - t2
    confirmed = ev.is_set()
    cleanup()
    tele.stop_rotation()
    tele.stop_tracking()

    with tele.lock:
        apex, low, rot_peak = tele.max_z, tele.min_z, tele.rot_max
    log.info('Entry: vz=%.2f m/s at z=%.2f m', entry_vz, entry_z)
    log.info('Rotation: handed over at %.0f deg after %.2f s%s, peak %.0f deg '
             '(overshoot past 360: %.0f deg)',
             handover_rot, rot_time, '  (TIMED OUT)' if timed_out else '',
             abs(rot_peak), max(0, abs(rot_peak) - 360))
    log.info('Recovery: %.2f s at full power', rec_time)
    log.info('Altitude: apex %.2f m, lowest %.2f m, lost %.2f m below the '
             'pre-flip height (estimator, see note)',
             apex, low, start_z - low)
    log.info('Note: the Lighthouse estimate can glitch while the deck is inverted; '
             'for the true loss, film the flip next to a tape mark on a wall.')
    if not confirmed:
        log.warning('Angle mode switch was not confirmed in time.')


def main(do_flip):
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
    tele = Telemetry()
    log_confs = []
    log.info('Connected. Flight CSV will be %s', LOG_FILE)
    try:
        vbat = read_battery(scf)
        log.info('Battery: %s V', vbat)
        if vbat is None or vbat < MIN_BATTERY_V:
            log.error('Battery unreadable or too low. Charge it first.')
            return

        set_param_sync(cf, 'flightmode.stabModePitch', 1)
        log.info('Waiting for the position estimate to settle...')
        reset_estimator(scf)
        log_confs = start_telemetry(scf, tele)
        time.sleep(0.5)
        if not tele.received:
            log.error('No telemetry received. Aborting.')
            return

        with tele.lock:
            sx, sy, sz = tele.x, tele.y, tele.z
        target_z = min(sz + CLIMB_ABOVE_START, MAX_Z)
        log.info('Start position: (%.2f, %.2f, %.2f) -> hover at z=%.2f',
                 sx, sy, sz, target_z)
        if target_z - sz < 0.5:
            log.error('Not enough room below MAX_Z. Lower the start point or raise MAX_Z.')
            return

        cf.supervisor.send_arming_request(True)
        time.sleep(1.0)
        cf.commander.send_setpoint(0, 0, 0, 0)       # unlock raw setpoints
        cf.commander.send_notify_setpoint_stop()
        time.sleep(0.2)

        hl = cf.high_level_commander
        tele.phase = 'takeoff'
        log.info('Taking off...')
        hl.takeoff(target_z, TAKEOFF_TIME)
        time.sleep(TAKEOFF_TIME + 0.5)
        hl.go_to(sx, sy, target_z, 0.0, 1.0)
        time.sleep(1.5)

        tele.phase = 'hover'
        hover = measure_hover_thrust(tele)
        z_now = tele.snapshot()[0]
        log.info('Hovering at z=%.2f, measured hover thrust: %s', z_now, hover)
        if z_now > MAX_Z + 0.2:
            log.warning('Too high! Landing without flipping.')
            do_flip = False
        if hover is None:
            log.warning('Could not measure hover thrust, using %s.', FALLBACK_HOVER_THRUST)
            hover = FALLBACK_HOVER_THRUST
        elif hover > MAX_THRUST / 1.3:
            log.warning('Hover thrust is close to max. Little power left to '
                        'recover from a flip (heavy deck / weak battery / brushed motors).')

        if do_flip:
            log.info('Flipping!')
            backflip(cf, tele, hover)
            # Hand straight to the position controller to catch the drone.
            tele.phase = 'catch'
            for _ in range(int(1.5 * RATE_HZ)):
                cf.commander.send_position_setpoint(sx, sy, target_z, 0.0)
                time.sleep(DT)
            cf.commander.send_notify_setpoint_stop()
            time.sleep(0.1)
        else:
            time.sleep(1.0)

        tele.phase = 'land'
        log.info('Landing...')
        hl.land(sz, 3.0)
        time.sleep(3.5)
        hl.stop()
    except KeyboardInterrupt:
        log.warning('Interrupted — stopping.')
    finally:
        try:
            cf.param.set_value('flightmode.stabModePitch', '1')
            cf.commander.send_stop_setpoint()
            cf.commander.send_notify_setpoint_stop()
            time.sleep(0.1)
            # Stop the log streams so they don't keep running on the drone
            # (that's what causes "Error no LogEntry to handle" next time).
            for lc in log_confs:
                try:
                    lc.stop()
                except Exception:
                    pass
            time.sleep(0.1)
        finally:
            tele.save(LOG_FILE)
            scf.close_link()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crazyflie backflip')
    parser.add_argument('--no-flip', action='store_true',
                        help='Take off, hover, measure thrust and land (test run).')
    args = parser.parse_args()
    main(do_flip=not args.no_flip)