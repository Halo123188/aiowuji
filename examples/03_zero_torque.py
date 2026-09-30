#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio

import numpy as np

from aiowuji import WujiController


async def main() -> int:
    parser = argparse.ArgumentParser(description="Enable with zero current so the hand can be moved by hand.")
    parser.add_argument("hand_sn", nargs="?")
    args = parser.parse_args()

    controller = WujiController(args.hand_sn)
    await controller.start()
    try:
        controller.switch("torque")
        controller.set_freq(10)
        while True:
            await controller.set("torque", np.zeros(controller.dof))
            print(np.round(np.rad2deg(controller.state["qpos"]), 1))
    finally:
        await controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
