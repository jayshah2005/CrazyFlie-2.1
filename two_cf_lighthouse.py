"""
Two Crazyflies on one Crazyradio, positioned by Lighthouse. Three phases:

    python two_cf_lighthouse.py connect   # phase 1: connect both, print deck/battery/position
    python two_cf_lighthouse.py hover     # phase 2: take off, hover, land
    python two_cf_lighthouse.py rings     # phase 3: two linked rings, counter-rotating

Ctrl+C at any point during flight -> both drones land (not a motor cut).

Prereqs: each drone has a unique radio address, both drones have the same
Lighthouse system config (geometry + calibration) written to them.
"""
import math
import signal
import sys
import threading
import time

import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from cflib.crazyflie.syncLogger import SyncLogger

# Same channel, different addresses -> one radio serves both links.
URIS = [
    'radio://0/80/2M/E7E7E7E7EA',
    'radio://0/80/2M/E7E7E7E7EB',
]

# Flight parameters (metres, seconds, Lighthouse frame)
FLIGHT_Z = 0.6        # hover / ring altitude
HOVER_TIME = 5.0      # phase 2 hover duration
R = 0.4               # ring radius; rings are tangent at the origin
LAP_PERIOD = 8.0      # seconds per lap (speed = 2*pi*R / period ~ 0.31 m/s)
LAPS = 3
SETPOINT_DT = 0.1     # 10 Hz position setpoint streaming

ABORT = threading.Event()


def on_sigint(sig, frame):
    print('\n[!] Ctrl+C: landing both drones')
    ABORT.set()


def wait(seconds):
    """Sleep that wakes early on abort. Returns True if aborted."""
    return ABORT.wait(seconds)


# ---------- phase 1 ----------
def check(scf):
    cf = scf.cf
    lh_deck = cf.param.get_value('deck.bcLighthouse4')
    flow_deck = cf.param.get_value('deck.bcFlow2')
    lg = LogConfig(name='state', period_in_ms=100)
    for v in ('pm.vbat', 'stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z'):
        lg.add_variable(v, 'float')
    with SyncLogger(scf, lg) as logger:
        for _, d, _ in logger:
            print(f"{cf.link_uri}: lighthouse={lh_deck} flow={flow_deck} "
                  f"vbat={d['pm.vbat']:.2f}V "
                  f"pos=({d['stateEstimate.x']:.2f}, {d['stateEstimate.y']:.2f}, "
                  f"{d['stateEstimate.z']:.2f})")
            break


# ---------- shared ----------
def arm(scf):
    # Crazyflie 2.1 Brushless needs an explicit arming request before motors spin.
    scf.cf.platform.send_arming_request(True)
    time.sleep(1.0)


def disarm(scf):
    scf.cf.platform.send_arming_request(False)


def land(scf):
    cf = scf.cf
    cf.commander.send_notify_setpoint_stop()   # hand control back to high-level commander
    cf.high_level_commander.land(0.0, 2.0)
    time.sleep(2.5)
    cf.high_level_commander.stop()


# ---------- phase 2 ----------
def hover(scf):
    hl = scf.cf.high_level_commander
    hl.takeoff(FLIGHT_Z, 2.0)
    if wait(2.5) or wait(HOVER_TIME):
        return land(scf)
    land(scf)


# ---------- phase 3 ----------
def ring_pos(cx, s, theta):
    # Ring A: cx=-R, s=+1 (CCW). Ring B: cx=+R, s=-1 (CW).
    # With the same theta, separation = 2R*sqrt(1 + sin^2 theta) >= 2R always,
    # and the two drones are never at the tangent point at the same time.
    return cx + R * math.cos(theta), s * R * math.sin(theta)


def rings(scf, cx, s, t0):
    cf = scf.cf
    hl = cf.high_level_commander
    omega = 2 * math.pi / LAP_PERIOD
    theta0 = math.pi / 2

    hl.takeoff(FLIGHT_Z, 2.0)
    if wait(2.5):
        return land(scf)

    x0, y0 = ring_pos(cx, s, theta0)
    hl.go_to(x0, y0, FLIGHT_Z, 0.0, 2.0)
    if wait(max(0.0, t0 - time.time())):     # both threads start the lap clock together
        return land(scf)

    t_end = t0 + LAPS * LAP_PERIOD
    while not ABORT.is_set() and time.time() < t_end:
        theta = theta0 + omega * (time.time() - t0)
        x, y = ring_pos(cx, s, theta)
        cf.commander.send_position_setpoint(x, y, FLIGHT_Z, 0.0)
        ABORT.wait(SETPOINT_DT)

    land(scf)


# ---------- main ----------
def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else 'connect'
    signal.signal(signal.SIGINT, on_sigint)
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')

    with Swarm(URIS, factory=factory) as swarm:
        print('Connected to both.')
        swarm.parallel_safe(check)
        if phase == 'connect':
            return

        print('Resetting estimators and waiting for convergence...')
        swarm.reset_estimators()
        swarm.parallel_safe(check)
        if ABORT.is_set():
            return

        swarm.parallel_safe(arm)
        try:
            if phase == 'hover':
                swarm.parallel_safe(hover)
            elif phase == 'rings':
                t0 = time.time() + 6.0   # covers takeoff + move to ring start
                swarm.parallel_safe(rings, args_dict={
                    URIS[0]: [-R, +1, t0],
                    URIS[1]: [+R, -1, t0],
                })
            else:
                print(f'Unknown phase: {phase}')
        finally:
            swarm.parallel_safe(disarm)


if __name__ == '__main__':
    main()