"""
Fly the learned "Learning to Fly in Seconds" policy (Eschmann et al., RA-L 2024)
on a Crazyflie running the learning_to_fly firmware (rlt.* parameters).

Port of learning-to-fly/scripts/trigger.py for our Lighthouse setup:
  - pre-flight checks: L2F firmware present, battery, estimator converged,
    yaw close to 0 (the policy always turns to yaw 0 in the LIGHTHOUSE frame)
  - heights are relative to the take-off point
  - every flight writes a CSV under logs/ via run_log

DEAD-MAN SWITCH: hold ENTER to fly. Release it and packets stop; the
firmware cuts the motors ~200 ms later while the learned policy is flying
(the drone drops). Press Q (with Enter released) to quit. Ctrl+C is an
emergency stop. An emergency-stop watchdog also cuts the motors within 1 s if
this script dies or the radio drops; power-cycle the drone before the next
flight (the watchdog stays latched until reboot).
The key is read with GetAsyncKeyState, so it is Windows-only and sees Enter
even when this console is not focused.

Targets the Crazyflie 2.1 Brushless firmware build (arming required, policy
trained on the identified brushless model).

Modes:
  takeoff_and_switch  stock PID takes off, policy takes over after --switch-after s
  hover_learned       policy does the take-off itself (ground effect: try 2nd)
  figure_eight        PID take-off, then policy tracks a figure eight
  hover_original      stock PID only (baseline for comparison)

Run `python rl_trigger.py --mode takeoff_and_switch` FIRST.
"""
import argparse
import ctypes
import math
import msvcrt
import struct
import sys
import threading
import time
from datetime import datetime

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.commander import META_COMMAND_CHANNEL
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort
from cflib.utils import uri_helper
from cflib.utils.reset_estimator import reset_estimator

import run_log

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7EA')
log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()
LOG_FILE = run_log.csv_path('flight', script=__file__, when=STARTED_AT)

SEND_PERIOD = 0.02        # s, 50 Hz packets while Enter is held (firmware timeout: 200 ms).
VK_RETURN = 0x0D
VK_Q = 0x51
LEARNED_PACKET = struct.pack('<B', 1)  # metaLearnedController; payload MUST be empty.

# stabilizer.controller value of the out-of-tree (learned) controller in the
# 2026.08-based firmware (ControllerType enum). The drone boots on PID.
CONTROLLER_OOT = 6

# rlt.wn values (learning_to_fly_controller/rl_tools_controller.c, enum Mode).
MODE_POSITION = 1
MODE_FIGURE_EIGHT = 4

# m1..m4: motor commands the policy computes (rltm); out1..out4: ratios actually
# sent to the ESCs (motor.m*); rpm1..rpm4: measured eRPM (65535 = invalid read).
EXTRA = ['stateEstimate.roll', 'stateEstimate.pitch', 'gyro.x', 'gyro.y', 'gyro.z',
         'motor.m1', 'motor.m2', 'motor.m3', 'motor.m4',
         'motor.m1_rpm', 'motor.m2_rpm', 'motor.m3_rpm', 'motor.m4_rpm']
COLUMNS = ['drone_ms', 'phase', 'x', 'y', 'z', 'yaw_deg', 'vbat',
           'policy_active', 'err_x', 'err_y', 'err_z', 'm1', 'm2', 'm3', 'm4',
           'roll_deg', 'pitch_deg', 'gyro_x', 'gyro_y', 'gyro_z',
           'out1', 'out2', 'out3', 'out4', 'rpm1', 'rpm2', 'rpm3', 'rpm4']


def key_down(vk):
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


class Telemetry:
    def __init__(self):
        self.rows = []
        self.phase = 'idle'
        self.latest = {}  # newest values from the other log blocks

    def a_cb(self, timestamp, data, logconf):
        b = self.latest
        self.rows.append([
            timestamp, self.phase,
            data['stateEstimate.x'], data['stateEstimate.y'], data['stateEstimate.z'],
            data['stateEstimate.yaw'], data['pm.vbat'], data['rltrp.sm'],
            b.get('rltte.x'), b.get('rltte.y'), b.get('rltte.z'),
            b.get('rltm.m1'), b.get('rltm.m2'), b.get('rltm.m3'), b.get('rltm.m4'),
        ] + [b.get(name) for name in EXTRA])

    def other_cb(self, timestamp, data, logconf):
        self.latest.update(data)

    def save(self, path):
        written = run_log.write_csv(path, COLUMNS, self.rows, script=__file__,
                                    generated_at=datetime.now().astimezone())
        log.info('Flight log written to %s (%s samples)', written, len(self.rows))

    def summarize(self):
        active = [r for r in self.rows if r[7] == 1 and r[8] is not None]
        if not active:
            log.info('The learned policy never had control of the motors.')
            return
        exy = [math.hypot(r[8], r[9]) for r in active]
        ez = [abs(r[10]) for r in active]
        log.info('Policy in control for %.1f s', len(active) * SEND_PERIOD)
        log.info('Position error under policy: xy mean %.3f m / max %.3f m, '
                 'z mean %.3f m / max %.3f m',
                 sum(exy) / len(exy), max(exy), sum(ez) / len(ez), max(ez))
        log.info('Battery: %.2f V -> %.2f V', self.rows[0][6], self.rows[-1][6])


def start_telemetry(scf, tele):
    a = LogConfig(name='L2FState', period_in_ms=20)
    for name in ('stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
                 'stateEstimate.yaw', 'pm.vbat'):
        a.add_variable(name, 'float')
    a.add_variable('rltrp.sm', 'uint8_t')
    b = LogConfig(name='L2FPolicy', period_in_ms=20)
    for name in ('rltte.x', 'rltte.y', 'rltte.z'):
        b.add_variable(name, 'float')
    for name in ('rltm.m1', 'rltm.m2', 'rltm.m3', 'rltm.m4'):
        b.add_variable(name, 'uint16_t')
    c = LogConfig(name='L2FAttitude', period_in_ms=20)
    for name in EXTRA[:5]:
        c.add_variable(name, 'float')
    d = LogConfig(name='L2FMotors', period_in_ms=20)
    for name in EXTRA[5:]:
        d.add_variable(name, 'uint16_t')
    for conf, cb in ((a, tele.a_cb), (b, tele.other_cb), (c, tele.other_cb), (d, tele.other_cb)):
        scf.cf.log.add_config(conf)
        conf.data_received_cb.add_callback(cb)
        conf.start()
    return [a, b, c, d]


def read_pose(scf, duration=0.5):
    """Average x, y, z, yaw and battery over `duration` seconds."""
    samples = []
    lg = LogConfig(name='Pose', period_in_ms=20)
    for name in ('stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
                 'stateEstimate.yaw', 'pm.vbat'):
        lg.add_variable(name, 'float')
    lg.data_received_cb.add_callback(lambda t, d, c: samples.append(d))
    scf.cf.log.add_config(lg)
    lg.start()
    time.sleep(duration)
    lg.stop()
    if not samples:
        return None
    keys = samples[0].keys()
    return {k.split('.')[1]: sum(s[k] for s in samples) / len(samples) for k in keys}


def set_param(cf, name, value, timeout=2.0):
    """Set a parameter and wait until the drone reports the new value."""
    end = time.time() + timeout
    while time.time() < end:
        cf.param.set_value(name, str(value))
        time.sleep(0.1)
        if abs(float(cf.param.get_value(name)) - float(value)) < 1e-4:
            log.info('  %s = %s', name, value)
            return
    raise RuntimeError(f'Could not set {name} to {value}')


def send_learned(cf):
    pk = CRTPPacket()
    pk.port = CRTPPort.COMMANDER_GENERIC
    pk.channel = META_COMMAND_CHANNEL
    pk.data = LEARNED_PACKET
    cf.send_packet(pk)


def configure(cf, args):
    log.info('Setting controller parameters:')
    set_param(cf, 'stabilizer.controller', CONTROLLER_OOT)  # learned controller (PID inside until triggered)
    set_param(cf, 'rlt.trigger', 0)  # policy runs while learned packets arrive
    if args.mode in ('takeoff_and_switch', 'hover_original'):
        set_param(cf, 'rlt.wn', MODE_POSITION)
        set_param(cf, 'rlt.target_z', 0)       # hold where the hand-over happens
        set_param(cf, 'rlt.motor_warmup', 0)
    elif args.mode == 'hover_learned':
        set_param(cf, 'rlt.wn', MODE_POSITION)
        set_param(cf, 'rlt.target_z', args.height)  # above the activation point
        set_param(cf, 'rlt.motor_warmup', 1)
    elif args.mode == 'figure_eight':
        set_param(cf, 'rlt.wn', MODE_FIGURE_EIGHT)
        set_param(cf, 'rlt.target_z_fe', 0)
        set_param(cf, 'rlt.fei', args.fe_interval)
        set_param(cf, 'rlt.fes', args.fe_scale)
        set_param(cf, 'rlt.motor_warmup', 0)


def preflight(scf, args):
    cf = scf.cf
    if cf.param.toc.get_element_by_complete_name('rlt.wn') is None:
        log.error('No rlt.* parameters: this drone is not running the '
                  'learning_to_fly firmware. Flash it first.')
        return None

    log.info('Waiting for the position estimate to converge ...')
    reset_estimator(scf)
    pose = read_pose(scf)
    if pose is None:
        log.error('No telemetry received.')
        return None
    log.info('Start pose: x=%.2f y=%.2f z=%.2f yaw=%.1f deg, battery %.2f V',
             pose['x'], pose['y'], pose['z'], pose['yaw'], pose['vbat'])

    if pose['vbat'] < args.min_vbat:
        log.error('Battery %.2f V is below %.2f V. Charge it: the policy does '
                  'not model battery sag.', pose['vbat'], args.min_vbat)
        return None
    if abs(pose['yaw']) > args.max_yaw_deg:
        log.error('Yaw is %.1f deg. The policy turns to yaw 0 of the Lighthouse '
                  'frame: rotate the drone to face +X (within %.0f deg) and retry.',
                  pose['yaw'], args.max_yaw_deg)
        return None
    return pose


def fly(scf, args, pose, tele):
    cf = scf.cf
    hover_z = pose['z'] + args.height
    uses_pid_takeoff = args.mode != 'hover_learned'

    while key_down(VK_RETURN):  # don't start on the Enter that launched us
        time.sleep(0.05)
    log.info('READY. Hold ENTER to fly (release = stop). Press Q to quit.')

    hold_start = None
    while not key_down(VK_Q):
        now = time.time()
        if not key_down(VK_RETURN):
            if hold_start is not None:
                log.warning('Enter released after %.1f s: packets stopped.',
                            now - hold_start)
            hold_start = None
            tele.phase = 'idle'
            time.sleep(SEND_PERIOD)
            continue

        if hold_start is None:
            hold_start = now
            log.info('Enter held: %s', 'PID take-off' if uses_pid_takeoff
                     else 'policy take-off')
        elapsed = now - hold_start
        use_pid = args.mode == 'hover_original' or (
            uses_pid_takeoff and elapsed < args.switch_after)

        if use_pid:
            tele.phase = 'pid'
            cf.commander.send_hover_setpoint(0, 0, 0, hover_z)
        else:
            if tele.phase != 'learned':
                log.info('Learned policy now in control.')
            tele.phase = 'learned'
            send_learned(cf)
        time.sleep(SEND_PERIOD)
    log.info('Q pressed: quitting.')


def main(args):
    if sys.platform != 'win32':
        log.error('The Enter dead-man switch uses the Windows key API.')
        return
    cflib.crtp.init_drivers()
    log.info('Connecting to %s ...', URI)
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        pose = preflight(scf, args)
        if pose is None:
            return
        configure(scf.cf, args)

        cf = scf.cf
        tele = Telemetry()
        confs = start_telemetry(scf, tele)

        # Emergency-stop watchdog: the firmware cuts the motors if these
        # keep-alives stop for 1 s (script crash, frozen laptop, lost radio).
        alive = threading.Event()
        alive.set()

        def keep_alive():
            while alive.is_set():
                cf.supervisor.send_emergency_stop_watchdog()
                time.sleep(0.2)

        threading.Thread(target=keep_alive, daemon=True).start()
        cf.supervisor.send_arming_request(True)  # the brushless firmware requires arming
        time.sleep(1.0)
        try:
            fly(scf, args, pose, tele)
        except KeyboardInterrupt:
            log.warning('Ctrl+C: EMERGENCY STOP, cutting motors.')
            cf.supervisor.send_emergency_stop()
        finally:
            alive.clear()
            cf.commander.send_stop_setpoint()
            cf.supervisor.send_arming_request(False)
            for conf in confs:
                conf.stop()
            tele.save(LOG_FILE)
            tele.summarize()
            # Enter is read from the keyboard directly, so the presses also queued
            # up in the console; drop them so they don't reach the shell.
            while msvcrt.kbhit():
                msvcrt.getwch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--mode', default='takeoff_and_switch',
                        choices=['takeoff_and_switch', 'hover_learned',
                                 'figure_eight', 'hover_original'])
    parser.add_argument('--height', type=float, default=0.3,
                        help='m above the take-off point (default 0.3)')
    parser.add_argument('--switch-after', type=float, default=3.0,
                        help='s of PID flight before the policy takes over')
    parser.add_argument('--fe-scale', type=float, default=0.3,
                        help='figure-eight scale; 1.0 = 2 m x 1 m (default 0.3)')
    parser.add_argument('--fe-interval', type=float, default=5.5,
                        help='s per figure-eight loop (paper: 15 / 5.5 / 3.5)')
    parser.add_argument('--max-yaw-deg', type=float, default=15.0,
                        help='refuse to fly if |yaw| is larger than this')
    parser.add_argument('--min-vbat', type=float, default=3.8,
                        help='refuse to fly below this battery voltage')
    main(parser.parse_args())
