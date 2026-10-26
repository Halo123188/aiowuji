#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio

from aiowuji import FINGERS, JOINT_LIMITS, JOINT_NAMES, WujiController


async def main() -> int:
    parser = argparse.ArgumentParser(description="Move one joint of each finger and back, finger by finger.")
    parser.add_argument("hand_sn", nargs="?")
    parser.add_argument("--slot", type=int, default=1, choices=range(1, 5),
                        help="which joint of each finger, S1-S4 (default 1)")
    parser.add_argument("--delta", type=float, default=1.0, help="rad (default 1.0, about 57 deg)")
    parser.add_argument("--max-velocity", type=float, default=2.0, help="rad/s (default 2.0)")
    parser.add_argument("--max-acceleration", type=float, default=10.0, help="rad/s^2 (default 10)")
    parser.add_argument("--max-jerk", type=float, default=100.0, help="rad/s^3 (default 100)")
    parser.add_argument("--cycles", type=int, default=3, help="out-and-back strokes per finger (default 3)")
    parser.add_argument("--disabled-joints", type=int, nargs="*", default=[],
                        help="joint indices (0-19) to leave disabled and skip, e.g. 17 for pinky_S2")
    args = parser.parse_args()

    # The defaults reach 2 rad/s in about 0.3 s: each 1.0 rad stroke takes
    # 0.8 s, about 0.3 s of it at full speed. Below roughly 0.5 rad/s the
    # joints stick and slip (see the README), so slow moves look stepped.
    speed = dict(max_velocity=args.max_velocity, max_acceleration=args.max_acceleration, max_jerk=args.max_jerk)
    joints = [finger * 4 + args.slot - 1 for finger in range(len(FINGERS))]
    joints = [joint for joint in joints if joint not in args.disabled_joints]

    controller = WujiController(args.hand_sn, disabled_joints=args.disabled_joints)
    await controller.start()
    try:
        q0 = controller.initial_qpos.copy()
        for joint in joints:
            low, high = JOINT_LIMITS[joint]
            if not low <= q0[joint] + args.delta <= high:
                raise ValueError(f"{JOINT_NAMES[joint]} would leave its range [{low:.2f}, {high:.2f}] rad")

        for joint in joints:
            print(f"{JOINT_NAMES[joint]}: {args.delta:+.2f} rad and back, {args.cycles}x")
            target = q0.copy()
            target[joint] += args.delta
            for _ in range(args.cycles):
                await controller.move(target, **speed)
                await controller.move(q0, **speed)
    finally:
        await controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
