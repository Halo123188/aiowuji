#!/usr/bin/env python3
"""Collect joint-impedance reference data for simulator system identification.

For every selected finger and every (kp, kd) pair, the finger's four joints
track one continuous trajectory, all moving at once. It has three segments,
each starting and ending at the finger's rest angles:

1. steps: every joint to the small step size and back, then to the large one;
2. slow sines: each joint at its own frequency, whole periods, peaking at the
   maximum amplitude (lower where the speed cap requires);
3. an interleaved multisine: each joint gets its own, non-overlapping set of
   harmonics (S1 gets 1, 5, 9, ...; S2 gets 2, 6, 10, ...), so the response of
   each joint and the coupling between them can be separated in the fit. It is
   scaled to the minimum amplitude and runs several whole periods.

The trajectories are planned against the bundled Wuji Hand 2 MuJoCo model
(aiowuji.model) so the finger never comes within --margin of another finger
or the palm. During a finger's trials the other fingers spread --spread away
from it (the thumb stays home), which gives its spread joint room. Each joint's
amplitude is first capped at 90% of its own collision-free range, then each
segment is scaled down as a whole wherever the combined motion would still
come too close. --plan prints the result.

The excited finger uses the trial gains; the others hold their pose with the
stiff reset gains. Before every trial the hand goes home and then to the
trial's clearance pose with the stiff gains (moves from the measured pose,
then a settle), so every trial starts from the same state. The default run
(5 fingers, 3 gain pairs) takes about 6 minutes.

qd_desired and effort_ff stay zero, so the reference is exactly the recorded
q_cmd. Each trial is saved as it finishes, under logs/sysid-<UTC time>/:

- trial_NNNN.npz: the per-tick recording (tick times, q_cmd sent, measured q,
  qd and effort, device seq and timestamps) for all 20 joints, plus the gains.
- manifest.json: hand, firmware, settings, and one entry per trial (finger,
  joints, gains, clearance pose, segments with their start times and
  per-joint parameters, headroom above the required model clearance, file,
  error after the reset).

Use --resume [DIR] to continue a stopped run (default: the newest
logs/sysid-*) with the same settings; the recorded trials are checked against
the plan first. Needs mujoco and tqdm: pip install "aiowuji[sysid]".
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from aiowuji import DOF, FINGERS, JOINT_LIMITS, JOINT_NAMES, WujiController

ROOT = Path(__file__).resolve().parents[1]
PRE, POST = 0.2, 0.3  # s recorded before and after each reference, holding the clearance pose
SETTLE = 0.25         # s held with the reset gains before a trial
RESET_S = 1.1         # measured time per reset (writes, moves, settle, save), for the estimate
STEP_HOLD = 0.4       # s at each step's target, then as long back
SINE_S = 8.0          # s of slow sines; joint j runs at (2 + j) / SINE_S Hz
MULTI_F0, MULTI_FMAX, MULTI_PERIODS = 0.25, 8.0, 3  # Hz, Hz, whole periods
FADE = 0.5            # s cosine fade at the ends of the sine and multisine segments
CHECK_HZ = 100        # trajectory samples per second checked against the model
LOWS, HIGHS = np.array(JOINT_LIMITS).T


def gains(text: str) -> tuple[float, float]:
    kp, kd = (float(v) for v in text.split(":"))
    return kp, kd


def fade(t: float, duration: float) -> float:
    """0 -> 1 over FADE at the start, 1 -> 0 over FADE at the end."""
    edge = min(t, duration - t, FADE) / FADE
    return 0.5 - 0.5 * math.cos(math.pi * max(edge, 0.0))


def finger_joints(finger: str) -> list[int]:
    return list(range(FINGERS.index(finger) * 4, FINGERS.index(finger) * 4 + 4))


def clearance_pose(finger: str, spread: float) -> np.ndarray:
    """Home, with the other fingers' spread joints turned away from `finger`."""
    q = np.zeros(DOF)
    position = FINGERS.index(finger)
    for other in range(1, len(FINGERS)):  # the thumb stays home
        if other != position:
            q[other * 4 + 1] = -spread if other < position else spread  # + spreads toward the pinky
    return q


def segments(args) -> list[dict]:
    """The segments before collision limits, with per-joint amplitudes."""
    out = [{"kind": "step", "amplitudes": [a] * 4, "duration": 2 * STEP_HOLD} for a in args.step_sizes]
    # Each joint's sine peaks at max_amplitude unless its speed, fade included,
    # would pass max_velocity; then its amplitude is scaled down to fit.
    frequencies = [(2 + j) / SINE_S for j in range(4)]
    grid = np.arange(0.0, SINE_S, 1e-3)
    envelope = np.array([fade(t, SINE_S) for t in grid])
    amplitudes = []
    for f in frequencies:
        speed = np.abs(np.gradient(envelope * np.sin(2 * np.pi * f * grid), 1e-3)).max()
        amplitudes.append(float(min(args.max_amplitude, args.max_velocity / speed)))
    out.append({"kind": "sines", "duration": SINE_S, "frequencies": frequencies, "amplitudes": amplitudes})
    n = int(round(MULTI_FMAX / MULTI_F0))
    out.append({"kind": "multisine", "duration": MULTI_PERIODS / MULTI_F0, "f0": MULTI_F0,
                "amplitudes": [args.min_amplitude] * 4,
                "harmonics": [list(range(j + 1, n + 1, 4)) for j in range(4)]})
    start = 0.0
    for segment in out:
        segment["start"] = start
        start += segment["duration"]
    return out


def segment_reference(segment: dict, max_velocity: float):
    """Offsets of the four joints from their rest angles (rad) at time t into the segment."""
    kind, duration = segment["kind"], segment["duration"]
    amplitude = np.array(segment["amplitudes"])
    if kind == "step":
        return lambda t: amplitude if t < duration / 2 else np.zeros(4)
    if kind == "sines":
        omega = 2 * np.pi * np.array(segment["frequencies"])
        return lambda t: fade(t, duration) * amplitude * np.sin(omega * t)
    if kind == "multisine":
        # Per joint: amplitudes ~ 1/f (a flat velocity spectrum) and Schroeder
        # phases for a low crest factor, scaled so the peak angle is the joint's
        # amplitude and the peak speed at most max_velocity.
        f0, grid = segment["f0"], np.arange(0.0, 1.0 / segment["f0"], 1e-3)
        parts, scales, phases = [], [], []
        for k, peak in zip(segment["harmonics"], amplitude):
            k = np.array(k)
            phase = -np.pi * np.arange(k.size) * (np.arange(k.size) - 1) / k.size
            shape = ((1.0 / k) * np.sin(2 * np.pi * f0 * np.outer(grid, k) + phase)).sum(axis=1)
            speed = np.abs(np.gradient(shape, 1e-3)).max()
            scale = min(peak / np.abs(shape).max(), max_velocity / speed)
            parts.append((2 * np.pi * f0 * k, scale / k, phase))
            scales.append(float(scale)); phases.append(phase.tolist())
        segment.update(scales=scales, phases=phases)
        return lambda t: fade(t, duration) * np.array(
            [(weight * np.sin(omega * t + phase)).sum() for omega, weight, phase in parts])
    raise ValueError(f"unknown segment {kind}")


def reference(trial: dict, max_velocity: float):
    """Offsets of the four joints from their rest angles (rad) at time t into the trajectory."""
    parts = [(s["start"], s["duration"], segment_reference(s, max_velocity)) for s in trial["segments"]]

    def trajectory(t: float) -> np.ndarray:
        for start, duration, part in parts:
            if start <= t < start + duration:
                return part(t - start)
        return np.zeros(4)
    return trajectory


def sampled(segment: dict, max_velocity: float) -> np.ndarray:
    """The joint offsets a segment passes through, (samples, 4), for checking."""
    if segment["kind"] == "step":  # the joints sweep from rest to the target and back
        return np.outer(np.linspace(0.0, 1.0, 11), segment["amplitudes"])
    part = segment_reference(segment, max_velocity)
    return np.array([part(t) for t in np.arange(0.0, segment["duration"], 1.0 / CHECK_HZ)])


def fit_finger(model, finger: str, args) -> dict:
    """Plan a finger's trajectory so it keeps --margin from the rest of the hand."""
    joints, base = finger_joints(finger), clearance_pose(finger, args.spread)
    margin = args.margin / 1000
    check = model.clearance(finger, base, margin)

    def slack(offsets: np.ndarray) -> float:
        worst = math.inf
        for row in offsets:
            q = base.copy()
            q[joints] += row
            worst = min(worst, check.slack(q))
        return worst

    # Each joint's own collision-free range from the clearance pose, within its limits.
    free = []
    for i, joint in enumerate(joints):
        sides = []
        for direction, room in ((-1, base[joint] - LOWS[joint]), (1, HIGHS[joint] - base[joint])):
            lo, hi = 0.0, room
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                offset = np.zeros(4)
                offset[i] = direction * mid
                lo, hi = (mid, hi) if slack(offset[None]) >= 0 else (lo, mid)
            sides.append(lo)
        free.append(sides)

    planned = segments(args)
    for segment in planned:
        caps = [neg_pos[1] if segment["kind"] == "step" else min(neg_pos) for neg_pos in free]
        segment["amplitudes"] = [float(min(a, 0.9 * cap)) for a, cap in zip(segment["amplitudes"], caps)]
        # Scale the whole segment down where the joints together still come too close.
        requested = list(segment["amplitudes"])
        lo, hi = 0.0, 1.0
        if slack(sampled(dict(segment), args.max_velocity)) < 0:
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                segment["amplitudes"] = [a * mid for a in requested]
                lo, hi = (mid, hi) if slack(sampled(dict(segment), args.max_velocity)) >= 0 else (lo, mid)
            segment["amplitudes"] = [a * lo for a in requested]
        segment["scale"] = hi if hi == 1.0 else lo
        segment["headroom_mm"] = round(1000 * slack(sampled(dict(segment), args.max_velocity)), 2)
    return {"signal": "finger", "base": base.tolist(), "free_range": free,
            "duration": planned[-1]["start"] + planned[-1]["duration"], "segments": planned}


def check_moves(model, poses: list, margin: float) -> None:
    """Every finger keeps its clearance while the hand moves home -> each pose."""
    home = np.zeros(DOF)
    checks = [model.clearance(finger, home, margin) for finger in FINGERS]
    for pose in poses:
        for s in np.linspace(0.0, 1.0, 21):
            q = s * np.asarray(pose)
            if min(c.slack(q) for c in checks) < 0:
                raise SystemExit(f"moving to the clearance pose {np.round(pose, 2).tolist()} comes within "
                                 f"{margin * 1000:g} mm in the model; lower --spread")


async def reset(controller: WujiController, base: np.ndarray, kp: float, kd: float) -> float:
    """Go home, then to the trial's clearance pose, with stiff gains; the error left (rad)."""
    # Pin the target to the measured pose first, so stiffening cannot yank a
    # joint that lagged its reference, then move from there.
    controller.set_now("q_desired", np.clip(controller.state["qpos"], LOWS, HIGHS))
    controller.set_gains(kp, kd)
    await controller.move(np.zeros(controller.dof))
    await controller.move(base)
    await controller.wait(SETTLE)
    return float(np.abs(controller.state["qpos"] - base).max())


def matches(planned, recorded) -> bool:
    """Whether a recorded value is the planned one (extra recorded keys allowed)."""
    if isinstance(planned, dict):
        return isinstance(recorded, dict) and all(matches(v, recorded.get(k)) for k, v in planned.items())
    if isinstance(planned, list):
        return isinstance(recorded, list) and len(recorded) == len(planned) and all(map(matches, planned, recorded))
    if isinstance(planned, float):
        return isinstance(recorded, (int, float)) and math.isclose(recorded, planned, abs_tol=1e-9)
    return planned == recorded


def same_trial(planned: tuple, recorded: dict) -> bool:
    """Whether a manifest entry is the trial the plan puts at its position."""
    finger, kp, kd, spec = planned
    return matches({"finger": finger, "kp": kp, "kd": kd, **spec}, recorded)


def save_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(path)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("hand_sn", nargs="?")
    parser.add_argument("--fingers", nargs="+", choices=FINGERS, default=list(FINGERS))
    parser.add_argument("--handedness", choices=("right", "left"), default="right",
                        help="which model to plan with; checked against the hand (default right)")
    parser.add_argument("--gains", nargs="+", type=gains, default=[(1.5, 0.03), (3.0, 0.05), (6.0, 0.1)],
                        metavar="KP:KD", help="trial gains (default 1.5:0.03 3:0.05 6:0.1)")
    parser.add_argument("--reset-gains", type=gains, default=(8.0, 0.15), metavar="KP:KD",
                        help="stiff gains for resets and the fingers not being excited (default 8:0.15)")
    parser.add_argument("--step-sizes", nargs="+", type=float, default=[0.05, 0.3], help="rad (default 0.05 0.3)")
    parser.add_argument("--min-amplitude", type=float, default=0.1, help="rad, multisine peak (default 0.1)")
    parser.add_argument("--max-amplitude", type=float, default=0.6, help="rad, slow sine peak (default 0.6)")
    parser.add_argument("--max-velocity", type=float, default=2.0,
                        help="rad/s cap on each joint's reference speed (default 2)")
    parser.add_argument("--margin", type=float, default=5.0,
                        help="mm the finger keeps from other fingers and the palm, in the model (default 5)")
    parser.add_argument("--spread", type=float, default=0.3,
                        help="rad the other fingers spread away from the excited one (default 0.3)")
    parser.add_argument("--effort-limit", type=float, default=0.5, help="A (default 0.5)")
    parser.add_argument("--plan", action="store_true", help="print the planned trajectories and time, then exit")
    parser.add_argument("--resume", nargs="?", const="latest", metavar="DIR",
                        help="continue a stopped run in DIR (default: the newest logs/sysid-*) with the same settings")
    args = parser.parse_args()

    try:
        from tqdm import tqdm
        from aiowuji.model import HandModel
        model = HandModel(args.handedness)
    except ImportError:
        parser.error('needs mujoco and tqdm: pip install "aiowuji[sysid]"')
    started = time.perf_counter()
    check_moves(model, [clearance_pose(f, args.spread) for f in args.fingers], args.margin / 1000)
    plans = {finger: fit_finger(model, finger, args) for finger in args.fingers}
    print(f"planned {len(plans)} finger trajectories against the {args.handedness} hand model "
          f"in {time.perf_counter() - started:.1f} s (margin {args.margin:g} mm, spread {args.spread:g} rad)")

    schedule = [(finger, kp, kd, plans[finger]) for finger in args.fingers for kp, kd in args.gains]
    out, manifest = None, None
    if args.resume:
        runs = sorted((ROOT / "logs").glob("sysid-*"))
        out = runs[-1] if args.resume == "latest" and runs else Path(args.resume).expanduser()
        if not (out / "manifest.json").exists():
            parser.error(f"no manifest.json in {out}")
        manifest = json.loads((out / "manifest.json").read_text())
        done = manifest["trials"]
        if len(done) > len(schedule) or not all(same_trial(planned, recorded)
                                                for planned, recorded in zip(schedule, done)):
            parser.error(f"{out} was recorded with different settings; pass the options of the original run")
        print(f"resuming {out.name}: {len(done)} of {len(schedule)} trials already recorded")
    first = len(manifest["trials"]) if manifest else 0
    total = sum(RESET_S + PRE + spec["duration"] + POST for _, _, _, spec in schedule[first:])
    print(f"{len(args.fingers)} fingers x {len(args.gains)} gain pairs = {len(schedule)} trials, "
          f"{len(schedule) - first} to go, about {total / 60:.1f} min")
    if args.plan:
        for finger, spec in plans.items():
            spread = {FINGERS[j // 4]: round(math.degrees(v)) for j, v in enumerate(spec["base"]) if v}
            print(f"  {finger}: others spread {spread} deg")
            for s in spec["segments"]:
                peaks = " ".join(f"{math.degrees(a):5.1f}" for a in s["amplitudes"])
                print(f"    {s['kind']:9s} {s['start']:4.1f}-{s['start'] + s['duration']:4.1f} s  S1-S4 peak "
                      f"[{peaks}] deg  scale {s['scale']:.2f}  headroom {s['headroom_mm']:.1f} mm")
        return 0

    if out is None:
        out = ROOT / "logs" / f"sysid-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        out.mkdir(parents=True)
    stiff_kp, stiff_kd = args.reset_gains
    controller = WujiController(args.hand_sn, kp=stiff_kp, kd=stiff_kd, effort_limit=args.effort_limit)
    await controller.start()
    try:
        handedness = controller.hand.handedness().get() if hasattr(controller.hand, "handedness") else None
        if handedness is not None and str(handedness).lower() != args.handedness:
            raise RuntimeError(f"planned for a {args.handedness} hand, but this hand is {handedness}; "
                               f"pass --handedness {str(handedness).lower()}")
        if manifest is None:
            manifest = {"hand_sn": controller.hand.serial_number,
                        "firmware": getattr(getattr(controller.hand, "info", None), "firmware_version", None),
                        "handedness": args.handedness,
                        "created": datetime.now(timezone.utc).isoformat(), "rate_hz": controller.rate_hz,
                        "transport": controller.transport.name, "effort_limit": args.effort_limit,
                        "reset_gains": list(args.reset_gains), "pre_s": PRE, "post_s": POST,
                        "max_velocity": args.max_velocity, "margin_mm": args.margin, "spread": args.spread,
                        "joint_names": list(JOINT_NAMES),
                        "settings": {k: v for k, v in vars(args).items() if k not in ("plan", "resume")},
                        "sessions": [], "trials": []}
        elif manifest["hand_sn"] != controller.hand.serial_number:
            raise RuntimeError(f"{out} was recorded on {manifest['hand_sn']}, not {controller.hand.serial_number}")
        manifest.setdefault("sessions", []).append({"started": datetime.now(timezone.utc).isoformat(),
                                                    "first_trial": first + 1,
                                                    "thread_priority": controller.thread_priority})
        save_manifest(out / "manifest.json", manifest)
        controller.set_freq(500)
        per_finger = len(args.gains)
        with tqdm(total=len(args.fingers), initial=first // per_finger, desc="fingers", unit="finger",
                  dynamic_ncols=True) as fingers_bar:
            for f, finger in enumerate(args.fingers):
                if (f + 1) * per_finger <= first:
                    continue  # recorded in an earlier session
                spec, joints = plans[finger], finger_joints(finger)
                with tqdm(total=per_finger, initial=max(0, first - f * per_finger), desc=f"  {finger}",
                          unit="gain", leave=False, dynamic_ncols=True) as gains_bar:
                    for g, (kp, kd) in enumerate(args.gains):
                        if f * per_finger + g < first:
                            continue
                        label = f"kp {kp:g} kd {kd:g}"
                        gains_bar.set_postfix_str(f"{label}, moving to the trial pose")
                        trial = json.loads(json.dumps(spec))  # the reference adds per-trial details
                        base = np.array(trial["base"])
                        reset_error = await reset(controller, base, stiff_kp, stiff_kd)
                        kp_vec, kd_vec = np.full(controller.dof, stiff_kp), np.full(controller.dof, stiff_kd)
                        kp_vec[joints], kd_vec[joints] = kp, kd
                        controller.set_gains(kp_vec, kd_vec)
                        trajectory = reference(trial, args.max_velocity)

                        controller.start_recording(seconds=PRE + trial["duration"] + POST + 1.0)
                        await controller.wait(PRE)
                        t0 = shown = time.perf_counter()
                        while (t := time.perf_counter() - t0) < trial["duration"]:
                            q = base.copy()
                            q[joints] += trajectory(t)
                            await controller.set("q_desired", q)
                            if time.perf_counter() - shown > 0.25:
                                gains_bar.set_postfix_str(f"{label}, {t:4.1f}/{trial['duration']:.1f} s")
                                shown = time.perf_counter()
                        await controller.set("q_desired", base)
                        await controller.wait(POST)
                        ticks = controller.stop_recording()

                        index = len(manifest["trials"]) + 1
                        name = f"trial_{index:04d}.npz"
                        np.savez_compressed(out / name, **ticks, kp=kp_vec, kd=kd_vec, joints=joints)
                        manifest["trials"].append({"file": name, "finger": finger, "joints": joints,
                                                   "joint_names": [JOINT_NAMES[j] for j in joints],
                                                   "kp": kp, "kd": kd, "reset_error_rad": reset_error, **trial})
                        save_manifest(out / "manifest.json", manifest)
                        tqdm.write(f"[{index}/{len(schedule)}] {finger} {label}, "
                                   f"reset error {np.rad2deg(reset_error):.2f} deg")
                        gains_bar.update(1)
                fingers_bar.update(1)
    finally:
        await controller.stop()
        print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
