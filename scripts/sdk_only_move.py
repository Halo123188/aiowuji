#!/usr/bin/env python3
"""Slow single-joint move using only the Wuji SDK, as Wuji's own examples do.

No aiowuji code is involved. This follows examples/python/wuji_hand_2/2.publish.py
from the Wuji SDK: set the effort limit and MIT gains (kp 3, kd 0.05, 1.5 A),
enable, wait for Enabled, then publish JointCommand(position, 0, 0) frames at
200 Hz along a straight-line ramp. It moves one joint out by --delta and back,
logs every joint_states frame and every command, and saves them under
logs/sdk-<UTC time>/ for comparison with scripts/log_move.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

if sys.platform == "darwin":  # the macOS port keeps SDK state here, as aiowuji does
    os.environ.setdefault("WUJI_MACOS_STATE_DIR", str(Path.home()))

from wuji_sdk import ConnectOptions, DeviceType, JointCommand, SdkManager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOTAL_JOINTS = 20
NAMES = [f"{f}_S{s}" for f in ("thumb", "index", "middle", "ring", "pinky") for s in range(1, 5)]
_THUMB = ((-68, 74), (-85, 40), (-60, 90), (-60, 90))
_FINGER = ((-60, 90), (-40, 40), (-60, 120), (-60, 90))
LIMITS = [(math.radians(a), math.radians(b)) for a, b in _THUMB + _FINGER * 4]


def dense_index(nid: int) -> int:
    bus, slot = divmod(nid, 5)
    return bus * 4 + slot - 1


class Log:
    def __init__(self, hand) -> None:
        self.sub = hand.joint_states().subscribe()
        self.states, self.commands = [], []

    def drain(self) -> list[float] | None:
        latest = None
        while (frame := self.sub.recv()) is not None:
            q, qd, effort = [math.nan] * 20, [math.nan] * 20, [math.nan] * 20
            for j in frame.joints:
                i = dense_index(j.nid)
                q[i], qd[i], effort[i] = j.position, j.velocity, j.effort
            self.states.append((time.perf_counter_ns(), frame.header.seq, frame.header.timestamp_us, q, qd, effort))
            latest = q
        return latest

    def send(self, publisher, positions) -> None:
        publisher.send([JointCommand(p, 0.0, 0.0) for p in positions])
        self.commands.append((time.perf_counter_ns(), list(positions)))

    def save(self, out: Path) -> None:
        s, c = self.states, self.commands
        np.savez_compressed(
            out / "sdk_log.npz",
            state_host_ns=np.array([r[0] for r in s], np.int64), seq=np.array([r[1] for r in s], np.int64),
            timestamp_us=np.array([r[2] for r in s], np.int64), q=np.array([r[3] for r in s], np.float32),
            qd=np.array([r[4] for r in s], np.float32), effort=np.array([r[5] for r in s], np.float32),
            command_host_ns=np.array([r[0] for r in c], np.int64), q_cmd=np.array([r[1] for r in c], np.float32))


def ramp(log, publisher, start, goal, duration, hz) -> None:
    """Vendor smooth_home pattern: linear interpolation, one frame per tick, deadline pacing."""
    steps, dt = max(1, int(duration * hz)), 1.0 / hz
    t0 = time.monotonic()
    for i in range(1, steps + 1):
        a = i / steps
        log.send(publisher, [s + (g - s) * a for s, g in zip(start, goal)])
        log.drain()
        wait = (t0 + dt * i) - time.monotonic()
        if wait > 0:
            time.sleep(wait)


def hold(log, publisher, pose, seconds, hz) -> None:
    ramp(log, publisher, pose, pose, seconds, hz)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--joint", type=int, default=8, help="dense joint index (default 8, middle_S1)")
    parser.add_argument("--delta", type=float, default=0.25, help="rad (default 0.25)")
    parser.add_argument("--speed", type=float, default=0.2, help="rad/s (default 0.2)")
    parser.add_argument("--hz", type=float, default=200.0, help="publish rate (default 200, as 2.publish.py)")
    args = parser.parse_args()

    manager = SdkManager.instance()
    hands = [d for d in manager.scan() if d.device_type == DeviceType.WujiHand2]
    if not hands:
        print("No Wuji Hand 2 found")
        return 1
    hand = manager.connect(sn=hands[0].sn, device_name="wuji_hand_2")
    out = ROOT / "logs" / f"sdk-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True)
    publisher, log = None, None
    try:
        if hand.online_joints_count().get() != TOTAL_JOINTS:
            raise RuntimeError("not all 20 joints are online")
        log = Log(hand)
        deadline = time.monotonic() + 2.0
        start = None
        while start is None and time.monotonic() < deadline:
            start = log.drain()
            time.sleep(0.01)
        if start is None:
            raise RuntimeError("no joint_states")
        goal = list(start)
        goal[args.joint] += args.delta
        low, high = LIMITS[args.joint]
        if not low <= goal[args.joint] <= high:
            raise RuntimeError(f"{NAMES[args.joint]} target outside its range")

        hand.effort_limit().set(1.5)
        hand.mit_params().set((3.0, 0.05))
        hand.enable()
        diag = hand.joint_diagnostics().subscribe()
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                time.sleep(0.2)
                frame = diag.recv()
                if frame is not None and frame.joints and all(e.status_word.ext_state == 2 for e in frame.joints):
                    break
            else:
                raise RuntimeError("enable timeout")
        finally:
            diag.close()
        publisher = hand.joint_command().publish()

        duration = abs(args.delta) / args.speed
        print(f"{NAMES[args.joint]}: {args.delta:+.2f} rad over {duration:.2f} s and back at {args.hz:.0f} Hz")
        hold(log, publisher, start, 0.5, args.hz)
        ramp(log, publisher, start, goal, duration, args.hz)
        hold(log, publisher, goal, 0.5, args.hz)
        ramp(log, publisher, goal, start, duration, args.hz)
        hold(log, publisher, start, 0.5, args.hz)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if publisher is not None:
            publisher.close()
        hand.disable()
        manager.disconnect_all()
        if log is not None:
            log.save(out)
            (out / "run.json").write_text(json.dumps({"args": vars(args), "joint": NAMES[args.joint],
                                                      "states": len(log.states), "commands": len(log.commands)},
                                                     indent=2) + "\n")
            print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
