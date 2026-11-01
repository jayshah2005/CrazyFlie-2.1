"""
System-identification flight for the Crazyflie 2.1 Brushless (stock PID only).

Takes off, hovers, does quick height steps so the motors change speed, lands,
and logs commanded motor values next to the RPM the ESCs report (bidirectional
DShot) at 100 Hz. The CSV feeds the brushless simulator model for retraining
the learned controller (motor lag and command-to-RPM mapping).

Ctrl+C at any time cuts the motors (the drone drops from <= 0.6 m). An
emergency-stop watchdog also cuts them within 1 s if this script dies or the
radio link drops. Power-cycle the drone before the next flight: the watchdog
stays latched until reboot.
Needs a charged battery, the Flow deck (or Lighthouse) for position, ~1.5 m x
1.5 m clear space.
"""
import argparse
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

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7EA')
log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()
LOG_FILE = run_log.csv_path('sysid', script=__file__, when=STARTED_AT)

HOVER_Z = 0.4      # m
STEP_DZ = 0.2      # m, size of the height steps
STEP_TIME = 0.4    # s, how fast each step is commanded (short = sharp motor change)
HOLD_TIME = 2.5    # s between steps
N_STEPS = 3
MAX_FLOOR_Z = 0.10  # m, pre-flight: |height| allowed while sitting on the floor
MAX_SPREAD = 0.05   # m, pre-flight: allowed estimate wobble while still

COLUMNS = ['drone_ms', 'phase', 'm1', 'm2', 'm3', 'm4',
           'rpm1', 'rpm2', 'rpm3', 'rpm4', 'vbat', 'z', 'vz']


class Telemetry:
    def __init__(self):
        self.rows = []
        self.phase = 'ground'
        self.latest_b = {}

    def a_cb(self, timestamp, data, logconf):
        b = self.latest_b
        self.rows.append([
            timestamp, self.phase,
            data['motor.m1'], data['motor.m2'], data['motor.m3'], data['motor.m4'],
            data['motor.m1_rpm'], data['motor.m2_rpm'], data['motor.m3_rpm'], data['motor.m4_rpm'],
            b.get('pm.vbat'), b.get('stateEstimate.z'), b.get('stateEstimate.vz'),
        ])

    def b_cb(self, timestamp, data, logconf):
        self.latest_b = data


def start_telemetry(scf, tele):
    a = LogConfig(name='SysIdMotors', period_in_ms=10)
    for i in range(1, 5):
        a.add_variable(f'motor.m{i}', 'uint16_t')
    for i in range(1, 5):
        a.add_variable(f'motor.m{i}_rpm', 'uint16_t')
    b = LogConfig(name='SysIdState', period_in_ms=10)
    for name in ('pm.vbat', 'stateEstimate.z', 'stateEstimate.vz'):
        b.add_variable(name, 'float')
    for conf, cb in ((a, tele.a_cb), (b, tele.b_cb)):
        scf.cf.log.add_config(conf)
        conf.data_received_cb.add_callback(cb)
        conf.start()
    return [a, b]


def fly(cf, tele):
    hlc = cf.high_level_commander
    tele.phase = 'takeoff'
    log.info('Taking off to %.1f m', HOVER_Z)
    hlc.takeoff(HOVER_Z, 2.0)
    time.sleep(3.0)
    tele.phase = 'hover'
    log.info('Hovering')
    time.sleep(4.0)
    for i in range(N_STEPS):
        tele.phase = f'step_up_{i}'
        hlc.go_to(0, 0, STEP_DZ, 0, STEP_TIME, relative=True)
        time.sleep(HOLD_TIME)
        tele.phase = f'step_down_{i}'
        hlc.go_to(0, 0, -STEP_DZ, 0, STEP_TIME, relative=True)
        time.sleep(HOLD_TIME)
    tele.phase = 'land'
    log.info('Landing')
    hlc.land(0.0, 2.0)
    time.sleep(2.5)
    hlc.stop()
    tele.phase = 'ground'


def summarize(rows):
    hover = [r for r in rows if r[1] == 'hover' and r[6]]
    if not hover:
        log.warning('No hover samples with RPM: is bidirectional DShot telemetry enabled?')
        return
    cmd = sum(sum(r[2:6]) / 4 for r in hover) / len(hover)
    rpm = sum(sum(r[6:10]) / 4 for r in hover) / len(hover)
    log.info('Hover: mean motor command %.0f / 65535 (%.1f%%), mean %.0f RPM, battery %.2f V',
             cmd, 100 * cmd / 65535, rpm, hover[-1][10])


def estimate_is_sane(scf, seconds=2.0):
    """Sample the position estimate on the floor; refuse to fly if it is off.

    A drone sitting still must read z close to 0 and a steady x/y. A jumping
    estimate (e.g. Lighthouse without a valid geometry) makes any controller
    fly away.
    """
    samples = []
    lg = LogConfig(name='PreflightPos', period_in_ms=20)
    for name in ('stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z'):
        lg.add_variable(name, 'float')
    lg.data_received_cb.add_callback(lambda t, d, c: samples.append(d))
    scf.cf.log.add_config(lg)
    lg.start()
    time.sleep(seconds)
    lg.stop()
    if len(samples) < 10:
        log.error('Pre-flight: no position data received.')
        return False
    spread = {}
    for axis in ('x', 'y', 'z'):
        v = [s[f'stateEstimate.{axis}'] for s in samples]
        spread[axis] = max(v) - min(v)
    z_mean = sum(s['stateEstimate.z'] for s in samples) / len(samples)
    log.info('Pre-flight estimate on the floor: z=%.3f m, spread x=%.3f y=%.3f z=%.3f m',
             z_mean, spread['x'], spread['y'], spread['z'])
    ok = True
    if abs(z_mean) > MAX_FLOOR_Z:
        log.error('Height reads %.2f m while on the floor (limit %.2f).',
                  z_mean, MAX_FLOOR_Z)
        ok = False
    for axis, value in spread.items():
        if value > MAX_SPREAD:
            log.error('Estimate %s moves %.3f m while the drone is still (limit %.3f).',
                      axis, value, MAX_SPREAD)
            ok = False
    if not ok:
        log.error('Not flying. Check the Lighthouse geometry / decks with still_test.py.')
    return ok



def main(args):
    cflib.crtp.init_drivers()
    log.info('Connecting to %s ...', URI)
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        if cf.param.get_value('stabilizer.controller') != '1':
            log.info('Selecting the PID controller')
            cf.param.set_value('stabilizer.controller', '1')
            time.sleep(0.5)
        log.info('Waiting for the position estimate to converge ...')
        reset_estimator(scf)
        if not estimate_is_sane(scf):
            return

        tele = Telemetry()
        confs = start_telemetry(scf, tele)
        time.sleep(0.5)
        vbat = tele.latest_b.get('pm.vbat')
        if vbat is not None and vbat < args.min_vbat:
            log.error('Battery %.2f V is below %.2f V. Charge it first.', vbat, args.min_vbat)
            for conf in confs:
                conf.stop()
            return

        # Emergency-stop watchdog: the firmware cuts the motors if these
        # keep-alives stop for 1 s (script crash, frozen laptop, lost radio).
        alive = threading.Event()
        alive.set()

        def keep_alive():
            while alive.is_set():
                cf.supervisor.send_emergency_stop_watchdog()
                time.sleep(0.2)

        threading.Thread(target=keep_alive, daemon=True).start()

        log.info('Arming (battery %.2f V). Ctrl+C cuts the motors.', vbat or float('nan'))
        cf.supervisor.send_arming_request(True)
        time.sleep(1.0)
        try:
            fly(cf, tele)
        except KeyboardInterrupt:
            log.warning('Ctrl+C: EMERGENCY STOP, cutting motors.')
            cf.supervisor.send_emergency_stop()
        finally:
            alive.clear()
            cf.commander.send_stop_setpoint()
            cf.supervisor.send_arming_request(False)
            for conf in confs:
                conf.stop()
            run_log.write_csv(LOG_FILE, COLUMNS, tele.rows, script=__file__,
                              generated_at=datetime.now().astimezone())
            log.info('Flight log written to %s (%s samples)', LOG_FILE, len(tele.rows))
            summarize(tele.rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--min-vbat', type=float, default=3.75,
                        help='refuse to fly below this battery voltage (default 3.75)')
    main(parser.parse_args())
