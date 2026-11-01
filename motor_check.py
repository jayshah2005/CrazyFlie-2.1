"""
Idle motor check for the Crazyflie 2.1 Brushless (props on, drone on the floor).

Arms the motors so they spin at idle (~2% of hover thrust: the drone cannot
lift off), measures each motor's eRPM for a few seconds, disarms, and repeats.
A motor far below the others (or at 0) means a motor/ESC/wiring problem; the
firmware's arming check will also disarm the drone in that case.

Usage: python motor_check.py [--rounds 3] [--seconds 3]
"""
import argparse
import statistics
import threading
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
LOG_FILE = run_log.csv_path('motors', script=__file__, when=STARTED_AT)

INVALID_RPM = 40000
LOW_FRACTION = 0.6  # flag a motor below 60% of the median idle RPM


def main(args):
    cflib.crtp.init_drivers()
    log.info('Connecting to %s ...', URI)
    rows = []
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        if cf.param.get_value('stabilizer.controller') != '1':
            cf.param.set_value('stabilizer.controller', '1')
            time.sleep(0.5)
        state = {'round': 0}
        lg = LogConfig(name='MotorCheck', period_in_ms=10)
        for i in range(1, 5):
            lg.add_variable(f'motor.m{i}', 'uint16_t')
        for i in range(1, 5):
            lg.add_variable(f'motor.m{i}_rpm', 'uint16_t')
        lg.data_received_cb.add_callback(lambda t, d, c: rows.append(
            [t, state['round']] + [d[f'motor.m{i}'] for i in range(1, 5)]
            + [d[f'motor.m{i}_rpm'] for i in range(1, 5)]))
        cf.log.add_config(lg)
        lg.start()

        alive = threading.Event()
        alive.set()

        def keep_alive():
            while alive.is_set():
                cf.supervisor.send_emergency_stop_watchdog()
                time.sleep(0.2)

        threading.Thread(target=keep_alive, daemon=True).start()
        try:
            for r in range(1, args.rounds + 1):
                state['round'] = r
                log.info('Round %d: arming (motors idle) for %.0f s', r, args.seconds)
                cf.supervisor.send_arming_request(True)
                time.sleep(args.seconds)
                cf.supervisor.send_arming_request(False)
                state['round'] = 0
                time.sleep(1.5)
        finally:
            alive.clear()
            cf.supervisor.send_arming_request(False)
            lg.stop()

    run_log.write_csv(LOG_FILE, ['drone_ms', 'round', 'm1', 'm2', 'm3', 'm4',
                                 'rpm1', 'rpm2', 'rpm3', 'rpm4'], rows,
                      script=__file__, generated_at=datetime.now().astimezone())
    log.info('Log written to %s', LOG_FILE)

    for r in range(1, args.rounds + 1):
        # skip the first 0.5 s of spin-up
        armed = [row for row in rows if row[1] == r][50:]
        spinning = [row for row in armed if any(row[2:6])]
        if not spinning:
            log.error('Round %d: motors never commanded (arming refused?)', r)
            continue
        rpm = []
        for i in range(4):
            vals = [row[6 + i] for row in spinning if row[6 + i] < INVALID_RPM]
            rpm.append(statistics.fmean(vals) if vals else 0.0)
        median = statistics.median(rpm)
        flags = ['LOW' if v < LOW_FRACTION * median else 'ok' for v in rpm]
        disarmed_early = len(spinning) < 0.8 * len(armed)
        log.info('Round %d: idle RPM M1..M4 = %s  %s%s', r,
                 ' '.join(f'{v:5.0f}' for v in rpm), ' '.join(flags),
                 '  (firmware cut the motors early)' if disarmed_early else '')
    log.info('Power-cycle the drone before flying: the emergency-stop watchdog '
             'stays latched until reboot.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=3.0)
    main(parser.parse_args())
