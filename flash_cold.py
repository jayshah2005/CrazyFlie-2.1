"""
Flash a firmware zip with a COLD boot only (never warm boot: `cfloader -w`
left this Crazyflie 2.1 Brushless dead once). Same as cfclient's "Cold boot
(recovery)" + "From file", for when cfclient cannot reach the bootloader.

Steps:
  1. Close cfclient (only one program can use the radio).
  2. Run: python flash_cold.py <firmware.zip>
  3. When asked: drone OFF, then hold the power button until the blue LEDs
     blink (bootloader mode). The script finds it and flashes the zip,
     including deck firmware (e.g. the Lighthouse FPGA) if needed.
  4. Power-cycle the drone afterwards.

Usage: python flash_cold.py <firmware.zip> [--wait 90]
"""
import argparse
import sys
import time
from datetime import datetime

import cflib.crtp
from cflib.bootloader import Bootloader

import run_log

log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()


def main(path, wait):
    cflib.crtp.init_drivers()
    bl = Bootloader(None)  # no URI: scan for a cold-booted bootloader
    log.info('Put the drone in bootloader mode NOW: power off, then hold the power '
             'button until the blue LEDs blink. Waiting up to %.0f s ...', wait)
    end = time.time() + wait
    started = False
    while time.time() < end and not started:
        try:
            started = bl.start_bootloader(warm_boot=False)
        except Exception as e:  # radio hiccups while scanning
            log.warning('scan error: %s', e)
            started = False
        if not started:
            time.sleep(0.5)
    if not started:
        log.error('No bootloader found. Is the blue LED blinking? Is cfclient closed?')
        return 1
    log.info('Bootloader found (protocol 0x%X). Flashing %s ...', bl.protocol_version, path)

    last = [None]

    def progress(msg, percent):
        key = (msg, percent // 10)
        if key != last[0]:
            log.info('%s %d%%', msg, percent)
            last[0] = key

    def info(_, target):
        log.info('Target: %s', target)

    # Empty target list = everything in the zip (as cfclient and cfloader do).
    bl.flash_full(cf=None, filename=path, warm=False, targets=[], info_cb=info, progress_cb=progress)
    bl.close()
    log.info('Done. Power-cycle the drone.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('zip', help='firmware zip (e.g. C:\\Users\\...\\l2f_firmware\\firmware-...zip)')
    parser.add_argument('--wait', type=float, default=90.0,
                        help='s to wait for the drone to enter bootloader mode')
    a = parser.parse_args()
    sys.exit(main(a.zip, a.wait))
