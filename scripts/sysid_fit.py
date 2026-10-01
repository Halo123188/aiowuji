#!/usr/bin/env python3
"""Fit MuJoCo joint dynamics to scripts/sysid_ref_collect.py recordings with mjbatch.

The model is the bundled Wuji Hand 2 MJCF with motor actuators: ctrl is motor
current (A) and each actuator's gear is its joint's torque constant (N m/A).
The firmware's PD runs on the host (aiowuji.sim.firmware_effort) at every
1 ms physics step, as the hand runs it. Its constants come from the data
first: a least-squares fit of the measured effort to kp (q_cmd - q) and
-kd qd over unsaturated samples gives kp_scale and kd_scale (the hand applies
about 1.95 times the documented law) and the lag, in 1 ms frames, between
sending a target and the hand using it.

Per finger, the fit then finds five parameters for each of its four joints,
shared across every gain setting:

- armature, damping, frictionloss: rotor inertia, viscous and Coulomb friction;
- stiffness, springref: a passive spring. With the hand fingers-up, gravity
  pulls a bent finger further, yet the measured effort pushes it back toward
  straight, so the joints carry a spring stronger than gravity.

They are searched in current units (A s^2/rad, A s/rad, A, A/rad) and
multiplied by the gear for the model. Motion without contact depends only on
those current-unit values; the gear itself is not identifiable from one mount,
because a linear spring absorbs gravity (they differ by sin q against q). It
defaults to the vendor's joint torque limits over the firmware's 2 A current
ceiling (--gear overrides). --fit-gear adds one scale per finger, which a
second recording with the hand in another orientation can pin down.

The search is CMA-ES, seeded by a least-squares inverse-dynamics fit of the
measured effort. Each candidate is scored on every trial of the finger, cut
into --window second windows that each start from the recorded state (so
stick-slip does not compound over a whole trial); the score is the RMS error
of the simulated joint angles. mjbatch steps all candidates x windows at once.
Contacts are off during the fit (the trajectories were planned 5 mm clear).

Writes, by default to <run>/fit/:

- wuji_hand2_<side>.xml: the full hand with the fitted joints, 1 ms
  implicitfast physics, motors with ctrlrange +-2 A, and the firmware
  constants as custom numerics; meshes/ is copied next to it. Load it with
  aiowuji.sim.WujiSim for a 50 Hz control loop over 1 kHz physics.
- params.json: the firmware law, every joint's parameters (model and current
  units, initial and fitted), and the error per finger, trial and joint, both
  per window and for a whole-trial rollout from the first recorded state.
- plots/<finger>.png: recorded vs simulated angles for each trial (whole-trial
  rollout), if matplotlib is installed.

Usage: sysid_fit.py [RUN[:MOUNT] ...] --mount fingers-up. RUN defaults to the
newest logs/sysid-*; MOUNT (fingers-up, fingers-down, palm-up, palm-down) or
--gravity X Y Z (m/s^2, wrist frame) sets gravity for runs without a suffix.
--holdout KP:KD leaves trials at those gains out of the fit and reports them
separately. Needs mjbatch (pip install mjbatch; it pins its own mujoco).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from aiowuji import FINGERS, JOINT_NAMES
from aiowuji.model import MODEL_DIR
from aiowuji.sim import firmware_effort, mjcf_joints

ROOT = Path(__file__).resolve().parents[1]
DT = 1e-3             # s per physics step and per telemetry frame
EFFORT_CEILING = 2.0  # A, the firmware's hard effort_limit ceiling: the motors' ctrlrange
MAX_LAG = 6           # frames of command lag tried
FRICTION_TIMECONST = 2 * DT  # s, joint solreffriction: MuJoCo clamps it to 2 timesteps
# Gravity in the wrist frame: the fingers extend along -z and curl toward +y (the palm).
MOUNTS = {"fingers-up": (0.0, 0.0, 9.81), "fingers-down": (0.0, 0.0, -9.81),
          "palm-up": (0.0, -9.81, 0.0), "palm-down": (0.0, 9.81, 0.0)}
# The vendor model's joint torque limits (N m) over EFFORT_CEILING: the default gear.
VENDOR_TORQUE = {"thumb": (0.6, 0.6, 0.3, 0.3), "finger": (2.0, 0.2, 0.3, 0.3)}
# Per joint, in current units: armature (A s^2/rad), damping (A s/rad),
# frictionloss (A), stiffness (A/rad) searched in log; springref (rad) linear.
PARAMS = ("armature", "damping", "frictionloss", "stiffness", "springref")
LOWER = np.array([np.log(2e-5), np.log(1e-5), np.log(1e-4), np.log(1e-4), -1.5])
UPPER = np.array([np.log(5e-2), np.log(0.5), np.log(0.5), np.log(3.0), 1.5])
STEP = np.array([0.7, 0.7, 0.7, 0.7, 0.2])  # initial search spread per parameter
GEAR_BOUND = math.log(10.0)                 # --fit-gear scales the gear within 1/10 to 10x
GEAR_STEP = 0.5
MIN_GENERATIONS = 50  # before --patience applies: the first ones explore far from the seed


def gains(text: str) -> tuple[float, float]:
    kp, kd = (float(v) for v in text.split(":"))
    return kp, kd


def default_gear(finger: str) -> np.ndarray:
    return np.array(VENDOR_TORQUE["thumb" if finger == "thumb" else "finger"]) / EFFORT_CEILING


@dataclass(eq=False)  # compared by identity
class Trial:
    """One trial of one finger on the hand's 1 ms telemetry frames (NaN where no frame arrived)."""

    file: str
    finger: str
    joints: list[int]
    kp: np.ndarray       # (4,) A/rad
    kd: np.ndarray       # (4,) A s/rad
    effort_limit: float  # A
    gravity: np.ndarray  # (3,) m/s^2 in the wrist frame
    q: np.ndarray        # (frames, 4) rad
    qd: np.ndarray       # (frames, 4) rad/s
    effort: np.ndarray   # (frames, 4) A
    tick: np.ndarray     # (frames,) the last loop tick that had seen each frame
    q_cmd: np.ndarray    # (ticks, 4) rad, the target each tick sent
    zoom: tuple[float, float]  # s from the first frame: 3 s of the multisine, for the plots

    def cmd(self, lag: int) -> np.ndarray:
        """The target the hand used at each frame: sent after it saw the frame `lag` earlier."""
        return self.q_cmd[self.tick[np.maximum(np.arange(len(self.tick)) - lag, 0)]]


def load_run(run: Path, gravity: np.ndarray) -> tuple[dict, list[Trial]]:
    manifest = json.loads((run / "manifest.json").read_text())
    trials = []
    for entry in manifest["trials"]:
        data, joints = np.load(run / entry["file"]), list(entry["joints"])
        multisine = next((s["start"] for s in entry["segments"] if s["kind"] == "multisine"), 0.0)
        zoom = manifest["pre_s"] + multisine + 1.0
        seq = data["seq"].astype(np.int64) - int(data["seq"][0])
        fresh = np.r_[True, np.diff(seq) > 0]
        arrays = {}
        for key in ("q", "qd", "effort"):
            arrays[key] = np.full((seq[-1] + 1, 4), np.nan)
            arrays[key][seq[fresh]] = data[key][fresh][:, joints]
        trials.append(Trial(
            file=f"{run.name}/{entry['file']}", finger=entry["finger"], joints=joints,
            kp=data["kp"][joints].astype(float), kd=data["kd"][joints].astype(float),
            effort_limit=float(manifest["effort_limit"]), gravity=np.asarray(gravity, float),
            tick=np.searchsorted(seq, np.arange(seq[-1] + 1), side="right") - 1,
            q_cmd=data["q_cmd"][:, joints].astype(float), zoom=(zoom, zoom + 3.0), **arrays))
    return manifest, trials


def firmware_law(trials: list[Trial]) -> dict:
    """kp_scale, kd_scale and lag from effort = kp_scale kp (q_cmd - q) - kd_scale kd qd."""
    by_lag = {}
    for lag in range(MAX_LAG + 1):
        rows, target = [], []
        for trial in trials:
            position, velocity = trial.kp * (trial.cmd(lag) - trial.q), -trial.kd * trial.qd
            nominal = 1.95 * position + 1.9 * velocity  # a rough guess, only to drop samples near the clip
            ok = (np.isfinite(trial.effort) & (np.abs(trial.effort) < 0.8 * trial.effort_limit)
                  & (np.abs(nominal) < 0.8 * trial.effort_limit))
            rows.append(np.c_[position[ok], velocity[ok]])
            target.append(trial.effort[ok])
        rows, target = np.concatenate(rows), np.concatenate(target)
        scale, *_ = np.linalg.lstsq(rows, target, rcond=None)
        by_lag[lag] = (scale, float(np.sqrt(np.mean((rows @ scale - target) ** 2))), len(target))
    lag = min(by_lag, key=lambda k: by_lag[k][1])
    (kp_scale, kd_scale), residual, samples = by_lag[lag]
    return {"kp_scale": float(kp_scale), "kd_scale": float(kd_scale), "lag_steps": int(lag),
            "residual_A": residual, "samples": samples,
            "residual_by_lag_A": {str(k): round(v[1], 5) for k, v in by_lag.items()}}


def build_spec(handedness: str, fingers, *, gear: dict, joints: dict | None = None, law: dict | None = None,
               contacts: bool = True, friction_timeconst: float = FRICTION_TIMECONST):
    """The vendor model with motor actuators, 1 ms implicitfast physics, and only `fingers`.

    gear and joints map aiowuji joint names to N m/A and to model-unit parameters;
    joints missing from `joints` keep the vendor's.
    """
    import mujoco

    spec = mujoco.MjSpec.from_file(str(MODEL_DIR / "mjcf" / f"{handedness}.xml"))
    spec.option.timestep = DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if not contacts:
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    names = dict(zip(mjcf_joints(handedness), JOINT_NAMES))
    for finger in FINGERS:
        if finger not in fingers:
            spec.delete(spec.joint(mjcf_joints(handedness)[FINGERS.index(finger) * 4]).parent)
    for actuator in list(spec.actuators):
        name = names[actuator.target]
        if name.split("_")[0] not in fingers:
            spec.delete(actuator)
            continue
        actuator.set_to_motor()
        actuator.gear[0] = gear[name]
        actuator.ctrllimited, actuator.ctrlrange = True, [-EFFORT_CEILING, EFFORT_CEILING]
        actuator.forcelimited = False
    for mjcf_name, name in names.items():
        if name.split("_")[0] not in fingers:
            continue
        joint = spec.joint(mjcf_name)
        joint.solref_friction = [friction_timeconst, 1.0]
        joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE  # the PD clips the current instead
        if joints and name in joints:
            p = joints[name]
            joint.armature, joint.frictionloss, joint.springref = p["armature"], p["frictionloss"], p["springref"]
            joint.damping, joint.stiffness = [p["damping"], 0.0, 0.0], [p["stiffness"], 0.0, 0.0]
    if law is not None:
        for key in ("kp_scale", "kd_scale", "lag_steps"):
            spec.add_numeric(name=f"aiowuji_{key}", data=[float(law[key])])
    return spec


def windows(trials: list[Trial], lag: int, length: int | None) -> dict:
    """Stack rollout windows of `length` frames (None: each whole trial), padded to one length.

    Each window starts from the first recorded frame at or after its start.
    """
    rows = []
    for trial in trials:
        frames = len(trial.q)
        span = frames - 1 if length is None else length
        cmd, measured = trial.cmd(lag), np.isfinite(trial.q[:, 0])
        for start in range(0, max(frames - span, 1), span):
            start = int(np.flatnonzero(measured[start:])[0]) + start
            stop = min(start + span, frames - 1)
            rows.append((trial, start, stop, cmd))
    steps = max(stop - start for _, start, stop, _ in rows)
    out = {k: [] for k in ("cmd", "meas", "q0", "qd0", "kp", "kd", "limit", "gravity")}
    for trial, start, stop, cmd in rows:
        pad = steps - (stop - start)
        out["cmd"].append(np.pad(cmd[start:stop], ((0, pad), (0, 0)), mode="edge"))
        out["meas"].append(np.pad(trial.q[start:stop + 1], ((0, pad), (0, 0)), constant_values=np.nan))
        qd0 = trial.qd[start]
        out["q0"].append(trial.q[start]); out["qd0"].append(np.where(np.isfinite(qd0), qd0, 0.0))
        out["kp"].append(trial.kp); out["kd"].append(trial.kd)
        out["limit"].append(trial.effort_limit); out["gravity"].append(trial.gravity)
    out = {k: np.array(v) for k, v in out.items()}
    out["valid"] = np.isfinite(out["meas"])
    out["meas"] = np.nan_to_num(out["meas"])
    out["steps"] = steps
    out["trial"] = np.array([next(i for i, u in enumerate(trials) if u is t) for t, *_ in rows])
    return out


class Rollouts:
    """Score parameter sets on a finger's windows, one simulation per (set, window)."""

    def __init__(self, model, law: dict, windows: dict) -> None:
        self.model, self.law, self.w = model, law, windows
        self.batches = {}

    def batch(self, sims: int):
        from mjbatch import Batch

        if sims not in self.batches:
            self.batches[sims] = Batch(self.model, sims)
        return self.batches[sims]

    def run(self, theta: np.ndarray, gear: np.ndarray, keep: bool = False):
        """theta (P, 4, 5) in current units, gear (P, 4) N m/A.

        Return the squared-error sums (P, windows) and, with keep, the angles (P, windows, steps + 1, 4).
        """
        w, law = self.w, self.law
        P, W = len(theta), len(w["cmd"])
        batch = self.batch(P * W)

        def per_sim(name: str) -> np.ndarray:
            field = batch.expand(name)
            return field.reshape(P, W, *field.shape[1:])

        physical = theta[..., :4] * gear[..., None]
        per_sim("dof_armature")[:] = physical[:, None, :, 0]
        per_sim("dof_damping")[:] = physical[:, None, :, 1]
        per_sim("dof_frictionloss")[:] = physical[:, None, :, 2]
        per_sim("jnt_stiffness")[:] = physical[:, None, :, 3]
        per_sim("qpos_spring")[:] = theta[:, None, :, 4]
        per_sim("actuator_gear")[..., 0] = gear[:, None]
        per_sim("gravity")[:] = w["gravity"][None]
        batch.set_const()
        batch.reset()
        q, qd = batch.bind("qpos").reshape(P, W, 4), batch.bind("qvel").reshape(P, W, 4)
        ctrl = batch.bind("ctrl").reshape(P, W, 4)
        q[:], qd[:] = w["q0"], w["qd0"]
        limit = w["limit"][:, None]
        sse = np.zeros((P, W))
        kept = np.empty((P, W, w["steps"] + 1, 4)) if keep else None
        if keep:
            kept[:, :, 0] = w["q0"]
        for s in range(w["steps"]):
            ctrl[:] = firmware_effort(w["cmd"][:, s], 0.0, q, qd, w["kp"], w["kd"], limit,
                                      law["kp_scale"], law["kd_scale"])
            batch.step()
            error = np.where(w["valid"][:, s + 1], q - w["meas"][:, s + 1], 0.0)
            sse += (error ** 2).sum(-1)
            if keep:
                kept[:, :, s + 1] = q
        sse[~np.isfinite(sse)] = np.inf
        return sse, kept


class CMA:
    """Plain CMA-ES (Hansen's tutorial defaults) minimizing over R^n from a mean of 0."""

    def __init__(self, n: int, popsize: int, seed: int) -> None:
        self.n, self.lam, self.mu = n, popsize, popsize // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.mean, self.sigma = np.zeros(n), 1.0
        self.pc, self.ps, self.C = np.zeros(n), np.zeros(n), np.eye(n)
        self.B, self.D = np.eye(n), np.ones(n)
        self.generation, self.rng = 0, np.random.default_rng(seed)

    def ask(self) -> np.ndarray:
        z = self.rng.standard_normal((self.lam, self.n))
        return self.mean + self.sigma * (z * self.D) @ self.B.T

    def tell(self, x: np.ndarray, cost: np.ndarray) -> None:
        best = x[np.argsort(cost)[: self.mu]]
        old, self.mean = self.mean, self.weights @ best
        y = (self.mean - old) / self.sigma
        inv_sqrt = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * inv_sqrt @ y
        self.generation += 1
        norm = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.generation))
        hsig = norm / self.chi < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        steps = (best - old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (steps.T * self.weights) @ steps)
        self.sigma *= math.exp(self.cs / self.damps * (np.linalg.norm(self.ps) / self.chi - 1))
        self.C = (self.C + self.C.T) / 2
        d2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(d2, 1e-20))


def smoothed_acceleration(q: np.ndarray, half: int = 10) -> np.ndarray:
    """Second derivative from a sliding quadratic fit over 2 half + 1 frames (NaNs interpolated)."""
    frames = np.arange(len(q))
    filled = np.column_stack([np.interp(frames, frames[np.isfinite(c)], c[np.isfinite(c)]) for c in q.T])
    k = np.arange(-half, half + 1)
    kernel = np.linalg.pinv(np.vander(k, 3, increasing=True))[2] * 2 / DT ** 2
    return np.column_stack([np.convolve(c, kernel[::-1], mode="same") for c in filled.T])


def inverse_dynamics(model, trials: list[Trial], gear: np.ndarray) -> np.ndarray:
    """Least-squares (4, 5) current-unit parameters from the measured effort while sliding."""
    from mjbatch import Batch

    samples = []
    for trial in trials:
        acc = smoothed_acceleration(trial.q)
        for frame in range(20, len(trial.q) - 20, 4):
            if np.all(np.isfinite(trial.q[frame])) and np.all(np.isfinite(trial.qd[frame])):
                samples.append((trial, frame, acc[frame]))
    batch = Batch(model, len(samples))
    batch.bind("qpos")[:] = [t.q[f] for t, f, _ in samples]
    batch.expand("gravity")[:] = [t.gravity for t, _, _ in samples]
    bias = batch.bind("qfrc_bias")
    batch.forward()
    holding = bias / gear  # A that holds each pose against gravity
    theta = np.empty((4, 5))
    for j in range(4):
        rows, target = [], []
        for (trial, frame, acc), hold in zip(samples, holding):
            v, e = trial.qd[frame, j], trial.effort[frame, j]
            if abs(v) > 0.1 and np.isfinite(e) and abs(e) < 0.9 * trial.effort_limit:
                rows.append((acc[j], v, np.sign(v), trial.q[frame, j], 1.0))
                target.append(e - hold[j])
        a, b, f, k, c = np.linalg.lstsq(np.array(rows), np.array(target), rcond=None)[0]
        k = max(k, math.exp(LOWER[3]))
        theta[j] = (a, b, f, k, -c / k)
    theta[:, :4] = np.clip(np.log(np.maximum(theta[:, :4], 1e-12)), LOWER[:4], UPPER[:4])
    theta[:, 4] = np.clip(theta[:, 4], LOWER[4], UPPER[4])
    return theta  # log for the first four, like the search


def decode(u: np.ndarray) -> np.ndarray:
    """(P, 4, 5) search coordinates (log, log, log, log, rad) -> current units."""
    u = np.clip(u, LOWER, UPPER)
    return np.concatenate([np.exp(u[..., :4]), u[..., 4:]], axis=-1)


def fit_finger(finger: str, trials: list[Trial], held_out: list[Trial], law: dict, args) -> dict:
    handedness = args.handedness
    names = [JOINT_NAMES[j] for j in trials[0].joints]
    base_gear = np.full(4, args.gear) if args.gear else default_gear(finger)
    spec = build_spec(handedness, [finger], gear=dict(zip(names, base_gear)), contacts=False,
                      friction_timeconst=args.friction_timeconst)
    model = spec.compile()
    assert list(model.jnt_qposadr) == [0, 1, 2, 3] and model.nu == 4

    u0 = inverse_dynamics(model, trials, base_gear)
    fit = Rollouts(model, law, windows(trials, law["lag_steps"], int(round(args.window / DT))))
    count = fit.w["valid"].sum()
    dims = 20 + bool(args.fit_gear)
    scale = np.r_[np.tile(STEP, 4), [GEAR_STEP] * bool(args.fit_gear)]

    def unpack(x):
        x = np.atleast_2d(x)
        u = u0 + (x[:, :20] * scale[:20]).reshape(-1, 4, 5)
        g = np.clip(x[:, 20] * GEAR_STEP, -GEAR_BOUND, GEAR_BOUND) if args.fit_gear else np.zeros(len(x))
        return decode(u), base_gear * np.exp(g)[:, None]

    def cost(x):
        theta, gear = unpack(x)
        sse, _ = fit.run(theta, gear)
        return np.sqrt(sse.sum(1) / count)

    from tqdm import tqdm

    started = time.perf_counter()
    initial = float(cost(np.zeros(dims))[0])
    cma = CMA(dims, args.popsize, args.seed + FINGERS.index(finger))
    best_x, best, stale = np.zeros(dims), initial, 0
    with tqdm(total=args.generations, desc=f"  {finger}", unit="gen", leave=False, dynamic_ncols=True) as bar:
        for _ in range(args.generations):
            x = cma.ask()
            c = cost(x)
            cma.tell(x, c)
            improved = c.min() < best * (1 - 1e-3)
            if c.min() < best:
                best, best_x = float(c.min()), x[np.argmin(c)].copy()
            stale = 0 if improved else stale + 1
            bar.set_postfix_str(f"RMS {1000 * best:.2f} mrad (initial {1000 * initial:.2f}), sigma {cma.sigma:.2f}")
            bar.update(1)
            if stale >= args.patience and cma.generation >= MIN_GENERATIONS:
                break
    theta, gear = (v[0] for v in unpack(best_x))
    theta0, gear0 = (v[0] for v in unpack(np.zeros(dims)))

    report = {"generations": cma.generation, "seconds": round(time.perf_counter() - started, 1),
              "window_rms_mrad": {"initial": round(1000 * initial, 3), "fitted": round(1000 * best, 3)},
              "trials": []}
    everything = trials + held_out
    whole = Rollouts(model, law, windows(everything, law["lag_steps"], None))
    sse, angles = whole.run(np.stack([theta0, theta]), np.stack([gear0, gear]), keep=True)
    per_window = Rollouts(model, law, windows(everything, law["lag_steps"], int(round(args.window / DT))))
    w_sse, w_angles = per_window.run(theta[None], gear[None], keep=True)
    for i, trial in enumerate(everything):
        valid = whole.w["valid"][i]
        rollout = np.sqrt(np.sum(np.where(valid, angles[:, i] - whole.w["meas"][i], 0) ** 2, axis=1) / valid.sum(0))
        mine = per_window.w["trial"] == i
        wvalid = per_window.w["valid"][mine]
        werr = np.where(wvalid, w_angles[0, mine] - per_window.w["meas"][mine], 0.0)
        windowed = np.sqrt((werr ** 2).sum(axis=(0, 1)) / wvalid.sum(axis=(0, 1)))
        report["trials"].append({"file": trial.file, "kp": float(trial.kp[0]), "kd": float(trial.kd[0]),
                                 "held_out": trial in held_out,
                                 "window_rms_mrad": np.round(1000 * windowed, 2).tolist(),
                                 "rollout_rms_mrad": np.round(1000 * rollout[1], 2).tolist(),
                                 "rollout_rms_mrad_initial": np.round(1000 * rollout[0], 2).tolist()})
    joints = {}
    for j, name in enumerate(names):
        current = dict(zip(PARAMS, theta[j].tolist()))
        joints[name] = {
            "mjcf_joint": mjcf_joints(handedness)[trials[0].joints[j]], "gear": float(gear[j]),
            **{p: current[p] * (1.0 if p == "springref" else float(gear[j])) for p in PARAMS},
            "current_units": current, "initial_current_units": dict(zip(PARAMS, theta0[j].tolist()))}
    if not args.no_plots:
        plot(finger, everything, held_out, whole, angles, Path(args.out) / "plots" / f"{finger}.png")
    return {"joints": joints, "report": report}


def plot(finger, trials, held_out, whole, angles, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 2 * len(trials), figsize=(5 * len(trials) * 2, 10), squeeze=False,
                             gridspec_kw={"width_ratios": [2, 1] * len(trials)})
    t = np.arange(whole.w["steps"] + 1) * DT
    for i, trial in enumerate(trials):
        shown = (t >= trial.zoom[0]) & (t <= trial.zoom[1])
        meas = np.where(whole.w["valid"][i], whole.w["meas"][i], np.nan)
        cmd = np.r_[whole.w["cmd"][i], whole.w["cmd"][i][-1:]]
        for j in range(4):
            for ax, span in ((axes[j, 2 * i], slice(None)), (axes[j, 2 * i + 1], shown)):
                ax.plot(t[span], cmd[span, j], color="0.75", lw=0.8, label="q_cmd")
                ax.plot(t[span], meas[span, j], color="k", lw=0.9, label="measured")
                ax.plot(t[span], angles[0, i, span, j], color="tab:orange", lw=0.7, alpha=0.8, label="initial guess")
                ax.plot(t[span], angles[1, i, span, j], color="tab:blue", lw=0.9, label="fitted")
            axes[j, 2 * i].set_ylabel(f"S{j + 1} (rad)")
        axes[0, 2 * i].set_title(f"{finger} kp {trial.kp[0]:g} kd {trial.kd[0]:g}"
                                 + (" (held out)" if trial in held_out else ""))
        axes[0, 2 * i + 1].set_title("multisine, zoomed")
        axes[-1, 2 * i].set_xlabel("s")
        axes[-1, 2 * i + 1].set_xlabel("s")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def parse_runs(items: list[str], default_gravity) -> list[tuple[Path, np.ndarray, str]]:
    runs = []
    for item in items or [None]:
        mount = None
        if item is None:
            found = sorted((ROOT / "logs").glob("sysid-*"))
            if not found:
                raise SystemExit("no logs/sysid-* runs; pass RUN")
            path = found[-1]
        else:
            path, _, mount = item.rpartition(":") if item.rpartition(":")[2] in MOUNTS else (item, "", None)
            path = Path(path).expanduser()
        if mount is not None:
            gravity, label = np.array(MOUNTS[mount]), mount
        elif default_gravity is not None:
            gravity, label = default_gravity
        else:
            raise SystemExit(f"how was the hand mounted for {path.name}? pass --mount or --gravity, or RUN:MOUNT")
        runs.append((path, gravity, label))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="*", metavar="RUN[:MOUNT]",
                        help="sysid run directories (default: the newest logs/sysid-*)")
    parser.add_argument("--mount", choices=sorted(MOUNTS), help="how the hand was mounted, for runs without :MOUNT")
    parser.add_argument("--gravity", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="gravity in the wrist frame (m/s^2), instead of --mount")
    parser.add_argument("--fingers", nargs="+", choices=FINGERS, default=list(FINGERS))
    parser.add_argument("--holdout", nargs="+", type=gains, default=[], metavar="KP:KD",
                        help="gain settings to leave out of the fit and report separately")
    parser.add_argument("--gear", type=float, help="torque constant (N m/A) for every joint "
                        "(default: the vendor's torque limits over 2 A)")
    parser.add_argument("--fit-gear", action="store_true", help="also fit one gear scale per finger")
    parser.add_argument("--window", type=float, default=2.0, help="s per rollout window (default 2)")
    parser.add_argument("--popsize", type=int, default=32, help="CMA-ES candidates per generation (default 32)")
    parser.add_argument("--generations", type=int, default=150, help="at most (default 150)")
    parser.add_argument("--patience", type=int, default=25,
                        help=f"stop after this many generations without a 0.1%% improvement, once "
                             f"{MIN_GENERATIONS} have run (default 25)")
    parser.add_argument("--friction-timeconst", type=float, default=FRICTION_TIMECONST,
                        help=f"joint solreffriction time constant (s, default {FRICTION_TIMECONST:g}: "
                             f"the stiffest MuJoCo allows at 1 ms steps; its own default 0.02 lets stuck joints creep)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="output directory (default: <first run>/fit)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    try:
        import mjbatch  # noqa: F401
        import mujoco  # noqa: F401
        import tqdm  # noqa: F401
    except ImportError as error:
        parser.error(f"needs mjbatch and tqdm ({error}): pip install mjbatch tqdm")

    default = None
    if args.gravity:
        default = (np.array(args.gravity), f"gravity {args.gravity}")
    elif args.mount:
        default = (np.array(MOUNTS[args.mount]), args.mount)
    runs = parse_runs(args.runs, default)
    manifests, trials = [], []
    for path, gravity, _ in runs:
        manifest, loaded = load_run(path, gravity)
        manifests.append(manifest)
        trials += [t for t in loaded if t.finger in args.fingers]
    if len({(m["hand_sn"], m["handedness"]) for m in manifests}) > 1:
        parser.error("the runs come from different hands")
    args.handedness = manifests[0]["handedness"]
    args.out = str(Path(args.out).expanduser() if args.out else runs[0][0] / "fit")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    law = firmware_law(trials)
    print(f"{len(trials)} trials from {len(runs)} run(s). Firmware PD from {law['samples']} unsaturated samples: "
          f"effort = clip({law['kp_scale']:.3f} kp (q_cmd - q) - {law['kd_scale']:.3f} kd qd), "
          f"target used {law['lag_steps']} ms after it is sent, residual {1000 * law['residual_A']:.1f} mA")

    def held(trial: Trial) -> bool:
        return any(math.isclose(trial.kp[0], kp) and math.isclose(trial.kd[0], kd) for kp, kd in args.holdout)

    results, gear_by_joint, joints = {}, {}, {}
    for finger in args.fingers:
        mine = [t for t in trials if t.finger == finger]
        fitted, held_out = [t for t in mine if not held(t)], [t for t in mine if held(t)]
        if not fitted:
            parser.error(f"every {finger} trial is held out")
        results[finger] = fit_finger(finger, fitted, held_out, law, args)
        report = results[finger]["report"]
        print(f"{finger}: window RMS {report['window_rms_mrad']['initial']:.2f} -> "
              f"{report['window_rms_mrad']['fitted']:.2f} mrad in {report['generations']} generations "
              f"({report['seconds']:.0f} s)")
        for t in report["trials"]:
            tag = " held out" if t["held_out"] else ""
            print(f"  kp {t['kp']:g} kd {t['kd']:g}{tag}: RMS per joint (mrad) windows {t['window_rms_mrad']}, "
                  f"whole trial {t['rollout_rms_mrad']} (initial guess {t['rollout_rms_mrad_initial']})")
        for name, p in results[finger]["joints"].items():
            joints[name], gear_by_joint[name] = p, p["gear"]

    # One model for the whole hand: fitted joints, and the vendor defaults for fingers not fitted.
    for finger in FINGERS:
        for name, g in zip([n for n in JOINT_NAMES if n.startswith(finger)],
                           np.full(4, args.gear) if args.gear else default_gear(finger)):
            gear_by_joint.setdefault(name, float(g))
    import mujoco

    fitted = [f for f in FINGERS if f in args.fingers]
    spec = build_spec(args.handedness, FINGERS, gear=gear_by_joint, joints=joints, law=law,
                      friction_timeconst=args.friction_timeconst)
    spec.modelname = f"wujihand2-{args.handedness}-sysid"
    meshes = out / "meshes" / args.handedness
    shutil.copytree(MODEL_DIR / "meshes" / args.handedness, meshes, dirs_exist_ok=True)
    spec.meshdir = f"{meshes.resolve()}/"  # to_xml compiles; the file refers to them relatively
    xml_path = out / f"wuji_hand2_{args.handedness}.xml"
    comment = (f"<!-- Fitted by aiowuji scripts/sysid_fit.py on {datetime.now(timezone.utc):%Y-%m-%d} from "
               f"{', '.join(p.name for p, _, _ in runs)} (fingers: {' '.join(fitted)}). Motors: ctrl is motor current "
               f"(A), gear is N m/A. Run the firmware PD on the host: see aiowuji.sim. -->\n")
    xml = spec.to_xml().replace(f'meshdir="{meshes.resolve()}/"', f'meshdir="meshes/{args.handedness}/"')
    xml_path.write_text(xml.replace("\n", "\n" + comment, 1) if xml.startswith("<mujoco") else comment + xml)
    mujoco.MjModel.from_xml_path(str(xml_path))  # it loads on its own

    summary = {
        "created": datetime.now(timezone.utc).isoformat(), "hand_sn": manifests[0]["hand_sn"],
        "firmware": manifests[0].get("firmware"), "handedness": args.handedness,
        "runs": [{"path": str(p), "gravity": g.tolist(), "mount": label} for p, g, label in runs],
        "model": xml_path.name, "firmware_law": law,
        "settings": {k: v for k, v in vars(args).items() if k not in ("runs", "handedness")},
        "units": {"model": "armature kg m^2, damping N m s/rad, frictionloss N m, stiffness N m/rad, "
                           "springref rad, gear N m/A",
                  "current_units": "the same divided by gear: A s^2/rad, A s/rad, A, A/rad"},
        "joints": joints, "fit": {f: r["report"] for f, r in results.items()}}
    (out / "params.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {xml_path} and {out / 'params.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
