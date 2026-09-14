#!/usr/bin/env python3
"""Run one short, bounded chassis pulse and print controller feedback."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comm.chassis import ChassisController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True)
    parser.add_argument('--speed', type=int, default=300)
    parser.add_argument('--duration', type=float, default=1.0)
    args = parser.parse_args()
    if not 0 < args.duration <= 2.0:
        raise ValueError('duration must be in (0, 2] seconds')

    chassis = ChassisController(port=args.port)
    chassis.connect()
    samples = []
    try:
        if not chassis.send_speed(args.speed, 0, 0):
            raise RuntimeError('serial write failed')
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            status = chassis.read_status()
            if status is not None:
                samples.append(status)
            time.sleep(0.05)
    finally:
        chassis.stop(repeat=5, interval=0.05)
        chassis.close()
    print(json.dumps({'samples': samples}, ensure_ascii=False))


if __name__ == '__main__':
    main()
