#!/usr/bin/env python3
"""Record the control loop during slow finger moves, to diagnose jerky motion.

This moves the hand like examples/00_move.py (one joint of each finger out and
back, finger by finger; slowly by default) with the controller's per-tick recording on,
and saves everything under logs/move-<UTC time>/:

- ticks.npz: for every 1 kHz loop tick, when it ran and when it was due, time
  spent sending, the command sent (q_cmd, qd_cmd, effort_cmd), and the newest
  joint_states (q, qd, effort, device seq and timestamp, host arrival time).
- run.json: settings, platform, thread priority, loop stats, and a summary.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from aiowuji import FINGERS, JOINT_LIMITS, JOINT_NAMES, WujiController

ROOT = Path(__file__).resolve().parents[1]


def percentiles(values, scale) -> dict:
    values = np.asarray(values, float) / scale
    if values.size == 0:
        return {}
    return {"p50": float(np.percentile(values, 50)), "p99": float(np.percentile(values, 99)),
            "max": float(values.max())}


def summarize(ticks: dict, moves: list[dict]) -> dict:
    tick_ns, seq = ticks["tick_ns"], ticks["seq"]
    seq_step = np.diff(seq)
    summary = {
        "ticks": int(tick_ns.size),
        "tick_interval_ms": percentiles(np.diff(tick_ns), 1e6),
        "ticks_over_2ms": int(np.sum(np.diff(tick_ns) > 2e6)),
        "tick_lateness_ms": percentiles(tick_ns - ticks["scheduled_ns"], 1e6),
        "send_ms": percentiles(ticks["send_ns"], 1e6),
        "telemetry_new_frames_per_tick": {str(k): int(np.sum(seq_step == k)) for k in (0, 1, 2)} |
                                         {"3+": int(np.sum(seq_step >= 3))},
        "telemetry_age_ms": percentiles(tick_ns - ticks["state_host_ns"], 1e6),
        "moves": [],
    }
    for move in moves:
        rows = (tick_ns >= move["start_ns"]) & (tick_ns <= move["end_ns"])
        j = move["joint"]
        error = ticks["q_cmd"][rows, j] - ticks["q"][rows, j]
        summary["moves"].append({
            "joint": JOINT_NAMES[j], "direction": move["direction"], "duration_s": move["duration_s"],
            "max_abs_error_rad": float(np.abs(error).max()) if error.size else None,
            "rms_error_rad": float(np.sqrt(np.mean(error ** 2))) if error.size else None,
        })
    return summary


def host_context() -> dict:
    context = {"platform": platform.platform(), "machine": platform.machine(),
               "python": sys.version.split()[0]}
    for name in ("aiowuji", "wuji-macos", "wuji-sdk", "ruckig", "numpy"):
        try:
            context[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    if sys.platform == "darwin":
        for key, command in (("power", ["pmset", "-g", "ps"]), ("pmset", ["pmset", "-g"])):
            try:
                context[key] = subprocess.run(command, capture_output=True, text=True, timeout=5).stdout
            except Exception as error:
                context[key] = f"unavailable: {error}"
    return context


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("hand_sn", nargs="?")
    parser.add_argument("--slot", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--delta", type=float, default=0.25, help="rad (default 0.25)")
    parser.add_argument("--max-velocity", type=float, default=0.2, help="rad/s (default 0.2)")
    parser.add_argument("--max-acceleration", type=float, default=0.5, help="rad/s^2 (default 0.5)")
    parser.add_argument("--max-jerk", type=float, default=2.0, help="rad/s^3 (default 2.0)")
    parser.add_argument("--kp", type=float, default=3.0, help="A/rad (default 3.0)")
    parser.add_argument("--kd", type=float, default=0.05, help="A*s/rad (default 0.05)")
    parser.add_argument("--friction", type=float, default=0.0,
                        help="friction feedforward while moving, A (default 0)")
    parser.add_argument("--hold", type=float, default=1.0, help="seconds held before and after (default 1)")
    parser.add_argument("--no-priority", action="store_true", help="leave the loop thread at default scheduling")
    parser.add_argument("--feed-forward", choices=("on", "off"),
                        help="set firmware joint feed-forward first (default: leave as is)")
    args = parser.parse_args()

    out = ROOT / "logs" / f"move-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True)
    speed = dict(max_velocity=args.max_velocity, max_acceleration=args.max_acceleration, max_jerk=args.max_jerk)
    joints = [finger * 4 + args.slot - 1 for finger in range(len(FINGERS))]
    run = {"args": vars(args), "speed": speed, "host": host_context(), "moves": []}

    feed_forward = None if args.feed_forward is None else args.feed_forward == "on"
    controller = WujiController(args.hand_sn, kp=args.kp, kd=args.kd, friction=args.friction,
                                realtime_priority=not args.no_priority, feed_forward=feed_forward)
    ticks, recording = None, False
    await controller.start()
    try:
        run.update(hand_sn=controller.hand.serial_number, transport=controller.transport.name,
                   thread_priority=controller.thread_priority,
                   firmware=getattr(controller.hand.info, "firmware_version", None))
        q0 = controller.initial_qpos.copy()
        for joint in joints:
            low, high = JOINT_LIMITS[joint]
            if not low <= q0[joint] + args.delta <= high:
                raise ValueError(f"{JOINT_NAMES[joint]} would leave its range [{low:.2f}, {high:.2f}] rad")

        controller.start_recording(seconds=120.0)
        recording = True
        await controller.wait(args.hold)
        for joint in joints:
            for direction, goal in (("out", q0.copy()), ("back", q0)):
                if direction == "out":
                    goal[joint] += args.delta
                print(f"{JOINT_NAMES[joint]} {direction}")
                start = time.perf_counter_ns()
                await controller.move(goal, **speed)
                end = time.perf_counter_ns()
                run["moves"].append({"joint": joint, "direction": direction, "start_ns": start,
                                     "end_ns": end, "duration_s": (end - start) / 1e9})
        await controller.wait(args.hold)
    finally:
        if recording:
            ticks = controller.stop_recording()
        stop_error = None
        try:
            await controller.stop()
        except Exception as error:  # save the log first; it matters most here
            stop_error = error
        run.update(stats=controller.stats, error=controller.error, warnings=sorted(controller.warnings),
                   stop_error=None if stop_error is None else str(stop_error),
                   joint_names=list(JOINT_NAMES))
        if ticks is not None:
            np.savez_compressed(out / "ticks.npz", **ticks)
            run["summary"] = summarize(ticks, run["moves"])
        (out / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n")
        print(f"\nSaved {out}")
        if "summary" in run:
            summary = run["summary"]
            print(f"loop: {summary['ticks']} ticks, interval {summary['tick_interval_ms']}, "
                  f"{summary['ticks_over_2ms']} over 2 ms; thread priority {controller.thread_priority}")
        if stop_error is not None:
            raise stop_error
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
