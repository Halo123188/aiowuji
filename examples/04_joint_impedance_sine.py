#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import time

import numpy as np

from aiowuji import JOINT_LIMITS, JOINT_NAMES, WujiController

SETTLE = 0.3  # s held at home after each joint's sine


def plot_response(ticks: dict, starts: list[int], period: float, title: str, path: Path) -> None:
    """One panel per joint: target and measured angle over that joint's own sine."""
    import matplotlib
    matplotlib.use("Agg")  # save only; no window
    import matplotlib.pyplot as plt  # pip install "aiowuji[plot]"

    fig, axes = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(12, 11),
                             layout="constrained", facecolor="#fcfcfb")
    for joint, (ax, start) in enumerate(zip(axes.flat, starts)):
        t = (ticks["tick_ns"] - start) / 1e9
        rows = (t >= -0.1) & (t <= period + SETTLE)
        target, measured = np.rad2deg(ticks["q_cmd"][rows, joint]), np.rad2deg(ticks["q"][rows, joint])
        ax.set_facecolor("#fcfcfb")
        ax.plot(t[rows], measured, color="#2a78d6", lw=2, label="measured")
        ax.plot(t[rows], target, color="#52514e", lw=1.5, ls=(0, (4, 3)), label="target", zorder=3)
        ax.set_title(f"{JOINT_NAMES[joint]}   max error {np.abs(target - measured).max():.1f}°",
                     loc="left", fontsize=10, color="#0b0b0b")
        ax.grid(color="#e1e0d9", lw=0.8)
        ax.tick_params(colors="#898781", labelcolor="#52514e")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
    for ax in axes[:, 0]:
        ax.set_ylabel("deg", color="#52514e")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", color="#52514e")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color="#0b0b0b")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper right", ncol=2,
               frameon=False, labelcolor="#0b0b0b")
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Excite each joint with one sine period in turn and plot its response.")
    parser.add_argument("hand_sn", nargs="?")
    parser.add_argument("--amplitude", type=float, default=0.3, help="rad around home (default 0.3, about 17 deg)")
    parser.add_argument("--frequency", type=float, default=1.0, help="Hz (default 1.0)")
    parser.add_argument("--kp", type=float, default=3.0, help="A/rad (default 3.0)")
    parser.add_argument("--kd", type=float, default=0.05, help="A*s/rad (default 0.05)")
    args = parser.parse_args()

    for joint, (low, high) in enumerate(JOINT_LIMITS):
        if not (low <= -args.amplitude and args.amplitude <= high):
            raise SystemExit(f"{JOINT_NAMES[joint]}: +/-{args.amplitude} rad leaves its range [{low:.2f}, {high:.2f}] rad")
    period, omega = 1.0 / args.frequency, 2.0 * np.pi * args.frequency

    controller = WujiController(args.hand_sn, kp=args.kp, kd=args.kd)
    await controller.start()
    try:
        q0 = np.zeros(controller.dof)  # the vendor home pose: all fingers straight
        await controller.move(q0)
        controller.set_freq(200)
        controller.start_recording(seconds=controller.dof * (period + SETTLE) + 1.0)
        starts = []
        for joint in range(controller.dof):
            print(f"{JOINT_NAMES[joint]}: +/-{args.amplitude:.2f} rad at {args.frequency:g} Hz")
            starts.append(time.perf_counter_ns())
            while (t := (time.perf_counter_ns() - starts[-1]) / 1e9) <= period:
                q = q0.copy()
                q[joint] += args.amplitude * np.sin(omega * t)
                await controller.set("q_desired", q)
            await controller.set("q_desired", q0)
            await controller.wait(SETTLE)
        ticks = controller.stop_recording()
    finally:
        await controller.stop()

    title = f"Sine response, kp {args.kp:g} A/rad, kd {args.kd:g} A·s/rad, ±{args.amplitude:g} rad at {args.frequency:g} Hz"
    path = (Path(__file__).resolve().parents[1] / "logs"
            / f"sine-kp{args.kp:g}-kd{args.kd:g}-{datetime.now():%Y%m%dT%H%M%S}.png")
    plot_response(ticks, starts, period, title, path)
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
