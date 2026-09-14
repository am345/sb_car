#!/usr/bin/env python3
"""Foreground constant-speed test; interrupt or lost feedback sends stop."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comm.chassis import ChassisController


def interrupted(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True)
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    chassis = ChassisController(port=args.port)
    chassis.connect()
    last_feedback = time.monotonic()
    last_print = 0.0
    ready = False
    try:
        while True:
            now = time.monotonic()
            status = chassis.read_status()
            if status is not None:
                last_feedback = now
                ready = True
                if status['flag_stop']:
                    raise RuntimeError('controller stop flag set')
                if now - last_print >= 1.0:
                    print(json.dumps({'command_x': 200, **status}), flush=True)
                    last_print = now
            if now - last_feedback > 1.0:
                raise RuntimeError('controller feedback timed out')
            if not chassis.send_speed(200 if ready else 0, 0, 0):
                raise RuntimeError('serial write failed')
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        chassis.stop(repeat=5, interval=0.05)
        chassis.close()
        print('STOPPED', flush=True)


if __name__ == '__main__':
    main()
