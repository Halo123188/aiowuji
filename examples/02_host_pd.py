#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio

import numpy as np

from aiowuji import WujiController


async def main() -> int:
    parser = argparse.ArgumentParser(description="Hold the initial pose with a PD law run in Python.")
    parser.add_argument("hand_sn", nargs="?")
    args = parser.parse_args()

    # Push the fingers around: the clip below (effort_limit, 0.5 A) bounds
    # how hard each joint pushes back.
    controller = WujiController(args.hand_sn)
    await controller.start()
    try:
        controller.switch("torque")
        controller.set_freq(500)

        kp, kd = 1.0, 0.02
        limit = controller.effort_limit
        q0 = controller.initial_qpos.copy()
        while True:
            state = controller.state
            error = q0 - state["qpos"]
            # Reported velocity can be nonzero while a joint is stationary;
            # keep kd small or filter a position difference instead.
            torque = kp * error - kd * state["qvel"]
            await controller.set("torque", np.clip(torque, -limit, limit))
    finally:
        await controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
