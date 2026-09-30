#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio

import numpy as np

from aiowuji import WujiController


async def main() -> int:
    parser = argparse.ArgumentParser(description="Hold the initial pose with firmware PD.")
    parser.add_argument("hand_sn", nargs="?")
    args = parser.parse_args()

    # Push the fingers around: effort_limit (0.5 A) bounds how hard each
    # joint pushes back.
    controller = WujiController(args.hand_sn)
    await controller.start()
    try:
        controller.switch("impedance")
        controller.kp = np.ones(controller.dof) * 1.0
        controller.kd = np.ones(controller.dof) * 0.05
        controller.set_freq(50)

        q0 = controller.initial_qpos.copy()
        while True:
            await controller.set("q_desired", q0)
    finally:
        await controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
