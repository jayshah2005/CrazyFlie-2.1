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
  land                as takeoff_and_switch, then after --hover-time s the policy's
                      target descends at --descent-rate; motors cut below
                      --cut-height (needs firmware with target-offset packets)
  hover_learned       policy does the take-off itself (ground effect: try 2nd)
  figure_eight        PID take-off, then policy tracks a figure eight
  hover_original      stock PID only (baseline for comparison)
  mission             no PID at all: the policy takes off from the floor, flies one
                      circle and lands, following a moving target (position and
                      velocity, needs the mission firmware and a mission-trained
                      policy). Releasing Enter in the air starts an automatic
                      landing (the policy descends at --abort-descent-rate);
                      Ctrl+C is the immediate motor cut. Q only quits on the floor.

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
           'out1', 'out2', 'out3', 'out4', 'rpm1', 'rpm2', 'rpm3', 'rpm4',
           'comp_x', 'comp_y', 'comp_z',  # mission: target correction (err_* is w.r.t. the corrected target)
           'lh_status', 'lh_receive', 'lh_active']  # Lighthouse: 2 = estimating; base-station bit masks


LH_VARS = ['lighthouse.status', 'lighthouse.bsReceive', 'lighthouse.bsActive']


def key_down(vk):
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


class Telemetry:
    def __init__(self):
        self.rows = []
        self.phase = 'idle'
        self.latest = {}  # newest values from the other log blocks
        self.comp = (0.0, 0.0, 0.0)  # mission target correction currently applied

    def a_cb(self, timestamp, data, logconf):
        b = self.latest
        self.rows.append([
            timestamp, self.phase,
            data['stateEstimate.x'], data['stateEstimate.y'], data['stateEstimate.z'],
            data['stateEstimate.yaw'], data['pm.vbat'], data['rltrp.sm'],
            b.get('rltte.x'), b.get('rltte.y'), b.get('rltte.z'),
            b.get('rltm.m1'), b.get('rltm.m2'), b.get('rltm.m3'), b.get('rltm.m4'),
        ] + [b.get(name) for name in EXTRA] + list(self.comp)
            + [b.get(name) for name in LH_VARS])

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
    e = LogConfig(name='L2FLighthouse', period_in_ms=20)
    toc = scf.cf.log.toc
    if toc.get_element_by_complete_name('lighthouse.status') is not None:
        e.add_variable('lighthouse.status', 'uint8_t')
        e.add_variable('lighthouse.bsReceive', 'uint16_t')
        e.add_variable('lighthouse.bsActive', 'uint16_t')
    confs = [(a, tele.a_cb), (b, tele.other_cb), (c, tele.other_cb), (d, tele.other_cb)]
    if e.variables:
        confs.append((e, tele.other_cb))
    for conf, cb in confs:
        scf.cf.log.add_config(conf)
        conf.data_received_cb.add_callback(cb)
        conf.start()
    return [conf for conf, _ in confs]


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


def send_learned(cf, offset=None, velocity=None):
    """Keep the policy flying; `offset` (x, y, z m) moves its target away from
    the activation point (needs firmware with target-offset support) and
    `velocity` (m/s) is the target's velocity (needs the mission firmware)."""
    pk = CRTPPacket()
    pk.port = CRTPPort.COMMANDER_GENERIC
    pk.channel = META_COMMAND_CHANNEL
    if offset is None:
        pk.data = LEARNED_PACKET
    elif velocity is None:
        pk.data = struct.pack('<Bfff', 1, *offset)
    else:
        pk.data = struct.pack('<Bffffff', 1, *offset, *velocity)
    cf.send_packet(pk)


class MissionProfile:
    """Target trajectory for --mode mission, relative to the take-off point.

    Mirrors MissionProfile in learning_to_fly/src/evaluate_robustness.cpp: sit,
    climb, hover, move to the circle start, circle, back to the centre, hover,
    descend to `below` m under the floor, sit."""
    PHASES = ['sit', 'takeoff', 'hover', 'move', 'circle', 'back', 'hover2', 'landing', 'ground']

    def __init__(self, args):
        self.H, self.R, self.P, self.laps = args.height, args.radius, args.period, args.laps
        self.v_up, self.v_down = args.climb_rate, args.descent_rate
        # sit only briefly: on the floor the policy rocks and slides the drone
        self.t_idle, self.t_hover1, self.t_move, self.t_back, self.t_hover2 = 0.3, 2.0, 2.0, 2.0, 1.0
        self.t_ground, self.below = 1.5, 0.02

    def duration(self):
        return (self.t_idle + self.H / self.v_up + self.t_hover1 + self.t_move + self.laps * self.P
                + self.t_back + self.t_hover2 + (self.H + self.below) / self.v_down + self.t_ground)

    def at(self, t):
        """(phase name, target offset (x, y, z), target velocity (x, y, z)) at time t."""
        H, R = self.H, self.R
        if t < self.t_idle:
            return 'sit', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        t -= self.t_idle
        t_up = H / self.v_up
        if t < t_up:
            return 'takeoff', (0.0, 0.0, self.v_up * t), (0.0, 0.0, self.v_up)
        t -= t_up
        if t < self.t_hover1:
            return 'hover', (0.0, 0.0, H), (0.0, 0.0, 0.0)
        t -= self.t_hover1
        if t < self.t_move:
            return 'move', (R * t / self.t_move, 0.0, H), (R / self.t_move, 0.0, 0.0)
        t -= self.t_move
        if t < self.laps * self.P:
            w = 2 * math.pi / self.P
            a = w * t
            return ('circle', (R * math.cos(a), R * math.sin(a), H),
                    (-R * w * math.sin(a), R * w * math.cos(a), 0.0))
        t -= self.laps * self.P
        if t < self.t_back:
            return 'back', (R * (1 - t / self.t_back), 0.0, H), (-R / self.t_back, 0.0, 0.0)
        t -= self.t_back
        if t < self.t_hover2:
            return 'hover2', (0.0, 0.0, H), (0.0, 0.0, 0.0)
        t -= self.t_hover2
        t_down = (H + self.below) / self.v_down
        if t < t_down:
            return 'landing', (0.0, 0.0, H - self.v_down * t), (0.0, 0.0, -self.v_down)
        return 'ground', (0.0, 0.0, -self.below), (0.0, 0.0, 0.0)


def configure(cf, args):
    log.info('Setting controller parameters:')
    set_param(cf, 'stabilizer.controller', CONTROLLER_OOT)  # learned controller (PID inside until triggered)
    set_param(cf, 'rlt.trigger', 0)  # policy runs while learned packets arrive
    if args.mode in ('takeoff_and_switch', 'land', 'hover_original', 'mission'):
        set_param(cf, 'rlt.wn', MODE_POSITION)
        set_param(cf, 'rlt.target_z', 0)       # target starts at the activation point
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
    if supervisor_locked(scf):
        log.error('The drone is locked (emergency stop from the previous run). '
                  'Power-cycle it and retry.')
        return None
    return pose


SUPERVISOR_LOCKED_BIT = 1 << 6  # supervisor.info bit layout in crazyflie-firmware


def supervisor_locked(scf):
    """True if the supervisor latched an emergency stop (only a reboot clears it)."""
    info = []
    lg = LogConfig(name='Supervisor', period_in_ms=100)
    lg.add_variable('supervisor.info', 'uint16_t')
    lg.data_received_cb.add_callback(lambda t, d, c: info.append(d['supervisor.info']))
    scf.cf.log.add_config(lg)
    lg.start()
    time.sleep(0.3)
    lg.stop()
    return bool(info and info[-1] & SUPERVISOR_LOCKED_BIT)


def fly_mission(scf, args, pose, tele):
    """--mode mission: the policy flies everything, following MissionProfile."""
    cf = scf.cf
    profile = MissionProfile(args)
    while key_down(VK_RETURN):  # don't start on the Enter that launched us
        time.sleep(0.05)
    log.info('Mission: climb to %.2f m at %.2f m/s, circle r=%.2f m in %.1f s x%g, land at '
             '%.2f m/s (%.0f s). No PID: the policy flies from the floor.',
             profile.H, profile.v_up, profile.R, profile.P, profile.laps, profile.v_down,
             profile.duration())
    log.info('READY. Hold ENTER for the whole mission. Releasing Enter in the air = '
             'automatic landing at %.2f m/s; Ctrl+C = cut the motors now. Q quits once '
             'landed.', args.abort_descent_rate)
    start = None
    done = False
    last_phase = None
    low_since = None
    abort_start = None   # Enter released in the air: automatic landing
    abort_floor = None
    last_cmd = None      # last commanded target offset (m)
    rows_seen = [-1, time.time()]  # telemetry sample count and when it last grew
    blind = False                  # telemetry lost: the automatic landing runs on the clock
    low_vbat_since = None
    # Outer integrator (--ki): the policy has no integral action and holds a
    # steady offset (~10 cm in simulation), so shift the commanded target to
    # cancel it. Runs while airborne (hover .. hover2); the height part fades
    # out near the floor so the landing target is unchanged.
    comp = list(args.comp_init)  # start from the last flight's correction
    if any(comp):
        log.info('Target correction starts at (%.2f, %.2f, %.2f) m (--comp-init).', *comp)
    last_tick = None
    while True:
        now = time.time()
        if done or start is None:
            # Q only quits on the floor (before the start or after landing): in the
            # air it would cut the motors. Ctrl+C is the emergency stop.
            if key_down(VK_Q):
                break
            if done or not key_down(VK_RETURN):
                time.sleep(SEND_PERIOD)
                continue
            start = now
            log.info('Enter held: mission started.')
        height = tele.rows[-1][4] - pose['z'] if tele.rows else 0.0
        # Safety 1: telemetry stopped (drone reset or power loss, or only the radio
        # downlink dropped). Do NOT stop sending: if our packets still get through,
        # stopping would cut the motors in the air (2026-11-04 1 m flight). Land
        # instead, timed by the clock since there is no height any more.
        if len(tele.rows) != rows_seen[0]:
            rows_seen[0], rows_seen[1] = len(tele.rows), now
        elif now - rows_seen[1] > 0.5 and not blind:
            blind = True
            if last_cmd is None or height < args.cut_height:
                log.error('No telemetry for %.1f s on the floor: stopping.', now - rows_seen[1])
                tele.phase = 'telemetry_lost'
                done = True
                continue
            log.error('No telemetry for %.1f s at %.2f m: the drone stopped answering '
                      '(power/battery or radio). Landing blind at %.2f m/s; Ctrl+C cuts the motors.',
                      now - rows_seen[1], height, args.abort_descent_rate)
            if abort_start is None:
                abort_start = now
        # Safety 2: battery sagging under load: land before it browns out.
        vbat = tele.rows[-1][6] if tele.rows else None
        if abort_start is None and vbat is not None and height > args.cut_height:
            if vbat < args.land_vbat:
                low_vbat_since = low_vbat_since or now
                if now - low_vbat_since > 0.5:
                    abort_start = now
                    log.warning('Battery %.2f V under load (< %.2f V) at %.2f m: AUTOMATIC '
                                'LANDING.', vbat, args.land_vbat, height)
            else:
                low_vbat_since = None
        if abort_start is None and not key_down(VK_RETURN):
            if last_phase == 'sit' or last_cmd is None or height < args.cut_height:
                log.warning('Enter released on the floor: motors cut, mission aborted.')
                tele.phase = 'aborted'
                done = True
                continue
            # Re-activating mid-air would restart the mission from there, so a
            # release in the air lands the drone instead (packets keep flowing).
            abort_start = now
            log.warning('Enter released %.1f s into the mission at %.2f m: AUTOMATIC '
                        'LANDING (Ctrl+C cuts the motors).', now - start, height)
        if abort_start is not None:
            z = max(-profile.below, last_cmd[2] - args.abort_descent_rate * (now - abort_start))
            vz = -args.abort_descent_rate if z > -profile.below else 0.0
            if z <= -profile.below and abort_floor is None:
                abort_floor = now
            tele.phase = 'autoland'
            if (z < 0.1 and height < args.cut_height) or \
                    (abort_floor is not None and now - abort_floor > args.floor_timeout):
                done = True
                tele.phase = 'landed'
                if blind:
                    log.info('Blind landing finished (no telemetry, height unknown): motors cut. Press Q.')
                else:
                    log.info('Landed (%.2f m above the take-off point): motors cut. Press Q.', height)
                continue
            send_learned(cf, (last_cmd[0], last_cmd[1], z), (0.0, 0.0, vz))
            time.sleep(SEND_PERIOD)
            continue
        phase, offset, velocity = profile.at(now - start)
        if phase != last_phase:
            log.info('Phase: %s', phase)
            last_phase = phase
        tele.phase = phase
        # Cut as soon as the drone is within --cut-height of the floor on the way
        # down: the simulator has no ground effect, and letting the policy settle
        # on the floor made it bounce (2026-11-04 0.6 m flight).
        if phase == 'ground':
            low_since = low_since or now
        if (phase == 'ground' or (phase == 'landing' and offset[2] < 0.1)) and height < args.cut_height \
                or (low_since is not None and now - low_since > args.floor_timeout):
            done = True
            tele.phase = 'landed'
            log.info('Landed (%.2f m above the take-off point): motors cut. '
                     'Release Enter and press Q.', height)
            continue
        dt = now - last_tick if last_tick is not None else 0.0
        last_tick = now
        # The height correction fades out near the floor. Before the first hover
        # (sit, take-off) the horizontal one fades in with height too, so the
        # target does not pull the drone sideways while it is on the floor.
        zs = min(1.0, max(0.0, offset[2] / profile.H))
        xs = zs if phase in ('sit', 'takeoff') else 1.0
        applied = (comp[0] * xs, comp[1] * xs, comp[2] * zs)
        err = [tele.latest.get(f'rltte.{a}') for a in 'xyz']  # commanded target - position
        if args.ki > 0 and phase in ('hover', 'move', 'circle', 'back', 'hover2') and None not in err:
            for i, k in enumerate((args.ki, args.ki, args.ki_z)):
                e = err[i] - applied[i]  # reference - position
                comp[i] = min(0.3, max(-0.3, comp[i] + k * e * dt))
            applied = (comp[0] * xs, comp[1] * xs, comp[2] * zs)
        tele.comp = applied
        last_cmd = tuple(o + c for o, c in zip(offset, applied))
        send_learned(cf, last_cmd, velocity)
        time.sleep(SEND_PERIOD)
    log.info('Q pressed: quitting. Final target correction (%.3f, %.3f, %.3f) m.', *comp)


def fly(scf, args, pose, tele):
    if args.mode == 'mission':
        fly_mission(scf, args, pose, tele)
        return
    cf = scf.cf
    hover_z = pose['z'] + args.height
    uses_pid_takeoff = args.mode != 'hover_learned'

    while key_down(VK_RETURN):  # don't start on the Enter that launched us
        time.sleep(0.05)
    log.info('READY. Hold ENTER to fly (release = stop). Press Q to quit.')

    hold_start = None
    learned_start = None
    landed = False
    while not key_down(VK_Q):
        now = time.time()
        if landed:
            time.sleep(SEND_PERIOD)
            continue
        if not key_down(VK_RETURN):
            if hold_start is not None:
                log.warning('Enter released after %.1f s: packets stopped.',
                            now - hold_start)
            hold_start = None
            learned_start = None
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
        elif args.mode != 'land':
            if tele.phase != 'learned':
                log.info('Learned policy now in control.')
            tele.phase = 'learned'
            send_learned(cf)
        else:
            if learned_start is None:
                learned_start = now
                log.info('Learned policy now in control; landing in %.0f s.', args.hover_time)
            descend_for = now - learned_start - args.hover_time
            dz = -min(args.height, max(0.0, descend_for * args.descent_rate))
            height = tele.rows[-1][4] - pose['z'] if tele.rows else args.height
            at_floor_for = descend_for - args.height / args.descent_rate
            if height < args.cut_height or at_floor_for > args.floor_timeout:
                landed = True
                tele.phase = 'landed'
                log.info('Landed (%.2f m above the take-off point): motors cut. '
                         'Release Enter and press Q.', height)
                continue
            if descend_for > 0 and tele.phase != 'landing':
                log.info('Descending at %.2f m/s.', args.descent_rate)
            tele.phase = 'landing' if descend_for > 0 else 'learned'
            send_learned(cf, (0.0, 0.0, dz))
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
                        choices=['takeoff_and_switch', 'land', 'hover_learned',
                                 'figure_eight', 'hover_original', 'mission'])
    parser.add_argument('--radius', type=float, default=0.3,
                        help='mission: circle radius in m (default 0.3)')
    parser.add_argument('--period', type=float, default=8.0,
                        help='mission: s per circle (default 8)')
    parser.add_argument('--laps', type=float, default=1.0,
                        help='mission: number of circles (default 1)')
    parser.add_argument('--climb-rate', type=float, default=0.25,
                        help='mission: take-off climb speed in m/s (default 0.25)')
    parser.add_argument('--ki', type=float, default=1.0,
                        help='mission: outer integrator gain in 1/s that cancels the '
                             'policy\'s steady offset (0 = off, default 1)')
    parser.add_argument('--land-vbat', type=float, default=3.45,
                        help='mission: land automatically if the battery stays below this '
                             'voltage under load for 0.5 s (default 3.45 V; brownout ~3.4 V)')
    parser.add_argument('--abort-descent-rate', type=float, default=0.3,
                        help='mission: descent speed of the automatic landing after Enter '
                             'is released in the air (m/s, default 0.3)')
    parser.add_argument('--ki-z', type=float, default=0.0,
                        help='mission: outer integrator gain for height (default 0: B2 '
                             'holds height well and the height correction chased the wobble)')
    parser.add_argument('--comp-init', type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=('X', 'Y', 'Z'),
                        help='mission: starting target correction in m (the "Final target '
                             'correction" of the last flight; the 2026-11-04 B2 flight '
                             'settled at 0.20 0.07 0)')
    parser.add_argument('--hover-time', type=float, default=5.0,
                        help='land: s of learned hover before descending')
    parser.add_argument('--descent-rate', type=float, default=0.1,
                        help='land, mission: target descent speed in m/s (default 0.1)')
    parser.add_argument('--cut-height', type=float, default=0.06,
                        help='land: cut motors below this height above take-off (m)')
    parser.add_argument('--floor-timeout', type=float, default=2.0,
                        help='land: cut motors this long after the target reached the floor')
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
