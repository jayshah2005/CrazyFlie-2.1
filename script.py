"""
Placeholder custom Crazyflie script.

Uses the shared run_log module so every console line and any CSV notes
which script produced it and when.
"""
from datetime import datetime

import run_log

log = run_log.setup(__file__)
STARTED_AT = datetime.now().astimezone()


def main():
    log.info('script.py is a stub — add your flight sequence here.')
    log.info('CSV files for this run would use: %s',
             run_log.csv_path('flight', script=__file__, when=STARTED_AT))
    log.info('Use run_log.write_csv(...) to save telemetry with script/time metadata.')


if __name__ == '__main__':
    main()
