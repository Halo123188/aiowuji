from __future__ import annotations

import asyncio
import contextlib
import ctypes
import os
import sys
import threading
import time

import numpy as np

from .hand import DOF, connect, import_sdk, outside_limits
from .transport import open_transport

# Upper bound for effort_limit (A): the value used by the vendor examples.
MAX_EFFORT_LIMIT = 1.5
SWITCH_INTERVAL_S = 0.0005
# A parameter write (gains, enable) holds the hand's joint_states back for up to
# about 20 ms (frames arrive late, none lost). While one is in flight and for
# WRITE_GRACE_S after it, telemetry may be up to WRITE_STALE_MS old.
WRITE_GRACE_S = 0.1
WRITE_STALE_MS = 150.0
_READY, _ENABLED, _STOPPED = 1, 2, 3
_TARGETS = ("q_desired", "qd_desired", "effort_ff", "torque")


class _TimeConstraint(ctypes.Structure):
    """Mach thread_time_constraint_policy_data_t; times in mach absolute units."""
    _fields_ = [("period", ctypes.c_uint32), ("computation", ctypes.c_uint32),
                ("constraint", ctypes.c_uint32), ("preemptible", ctypes.c_int)]


class _Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


def _mach_realtime(libc, period_s: float) -> bool:
    """Put the calling thread in the Mach real-time band, as Core Audio threads are.

    It declares a period, a CPU budget per period (half of it) and a deadline
    (90% of it); the kernel schedules it ahead of every QoS class and demotes
    it to normal priority if it overruns the budget. No root needed.
    """
    timebase = _Timebase()
    libc.mach_timebase_info(ctypes.byref(timebase))
    ticks = lambda seconds: int(seconds * 1e9 * timebase.denom / timebase.numer)
    policy = _TimeConstraint(ticks(period_s), ticks(0.5 * period_s), ticks(0.9 * period_s), 1)
    libc.pthread_self.restype = ctypes.c_void_p
    libc.pthread_mach_thread_np.argtypes = [ctypes.c_void_p]
    libc.pthread_mach_thread_np.restype = ctypes.c_uint32
    libc.thread_policy_set.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    THREAD_TIME_CONSTRAINT_POLICY, COUNT = 2, 4
    thread = libc.pthread_mach_thread_np(libc.pthread_self())
    return libc.thread_policy_set(thread, THREAD_TIME_CONSTRAINT_POLICY, ctypes.byref(policy), COUNT) == 0


def _prioritize_current_thread(period_s: float) -> str:
    """Give the calling thread the best scheduling the OS allows without setup.

    macOS: QoS class USER_INTERACTIVE (performance cores, minimal timer
    coalescing), then the Mach real-time policy for a loop of `period_s`.
    Linux: SCHED_FIFO priority 80 when permitted (root or CAP_SYS_NICE).
    Returns what was applied.
    """
    applied = "default"
    if sys.platform == "darwin":
        QOS_CLASS_USER_INTERACTIVE = 0x21
        libc = ctypes.CDLL(None)
        if libc.pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) == 0:
            applied = "qos_user_interactive"
        if _mach_realtime(libc, period_s):
            applied = "mach_realtime"
    elif sys.platform.startswith("linux"):
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(80))
            applied = "sched_fifo_80"
        except (PermissionError, OSError):
            pass
    return applied


class _Recording:
    """Per-tick record of the control loop, written by the loop thread."""

    def __init__(self, capacity: int) -> None:
        self.n = 0
        self.tick_ns = np.zeros(capacity, np.int64)       # when the tick ran
        self.scheduled_ns = np.zeros(capacity, np.int64)  # when it was due
        self.send_ns = np.zeros(capacity, np.int64)       # time spent in send
        self.state_host_ns = np.zeros(capacity, np.int64)  # when the newest state arrived
        self.seq = np.zeros(capacity, np.int64)
        self.timestamp_us = np.zeros(capacity, np.int64)
        self.q_cmd, self.qd_cmd, self.effort_cmd, self.q, self.qd, self.effort = (
            np.zeros((capacity, DOF), np.float32) for _ in range(6))

    def append(self, tick_ns, scheduled_ns, send_ns, command, state) -> None:
        i = self.n
        if i == len(self.tick_ns):
            return
        self.tick_ns[i], self.scheduled_ns[i], self.send_ns[i] = tick_ns, scheduled_ns, send_ns
        self.state_host_ns[i], self.seq[i], self.timestamp_us[i] = (
            state["host_ns"], state["seq"], state["timestamp_us"])
        self.q_cmd[i], self.qd_cmd[i], self.effort_cmd[i] = command
        self.q[i], self.qd[i], self.effort[i] = state["qpos"], state["qvel"], state["effort"]
        self.n = i + 1

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: value[:self.n] for name, value in vars(self).items() if name != "n"}


def _as_vector(value, name: str, *, nonnegative: bool = False) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        arr = np.full(DOF, float(arr))
    if arr.shape != (DOF,):
        raise ValueError(f"{name} must have shape {(DOF,)}, got {arr.shape}")
    if not np.all(np.isfinite(arr)) or (nonnegative and np.any(arr < 0)):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return arr.copy()


class WujiController:
    """Async joint controller for the Wuji Hand 2.

    Supported modes:
    - `impedance`: the hand firmware's PD (MIT mode) tracks `q_desired`:
      `effort = kp * (q_desired - q) + kd * (qd_desired - qd) + effort_ff`
      (plus the optional `friction` feedforward while the target moves)
    - `torque`: firmware gains are zero and `torque` is sent directly

    Efforts are motor currents in amperes (the SDK's Kt is 1). A background
    thread streams the command and checks telemetry at `rate_hz`. It disables
    the hand when joint_states go stale, a joint drops out or faults, or, if
    `max_tracking_error` is set (rad; off by default), the impedance tracking
    error exceeds it. Firmware warnings such as Stall are collected in
    `warnings` instead.
    """

    def __init__(
        self,
        hand=None,
        *,
        kp=3.0,
        kd=0.05,
        friction=0.0,
        effort_limit: float = 0.5,
        rate_hz: float = 1000.0,
        max_tracking_error: float | None = None,
        stale_ms: float = 20.0,
        transport: str = "auto",
        dry_run: bool = False,
        realtime_priority: bool = True,
        device_name: str = "aiowuji",
        feed_forward: bool | None = None,
    ) -> None:
        if not 0.0 <= effort_limit <= MAX_EFFORT_LIMIT:
            raise ValueError(f"effort_limit must be within [0, {MAX_EFFORT_LIMIT}] A")
        if not 100.0 <= rate_hz <= 1000.0:
            raise ValueError("rate_hz must be within [100, 1000]")
        self._hand_arg = hand
        self._kp = _as_vector(kp, "kp", nonnegative=True)
        self._kd = _as_vector(kd, "kd", nonnegative=True)
        self.effort_limit = float(effort_limit)
        self.friction = friction
        # Commanded speed (rad/s) at which the friction feedforward reaches
        # about 76% of its value (tanh(1)); it is zero when the target is still.
        self.friction_velocity = 0.02
        self.rate_hz = float(rate_hz)
        self.max_tracking_error = None if max_tracking_error is None else float(max_tracking_error)
        self.stale_ms = float(stale_ms)
        self.dry_run = bool(dry_run)
        self.realtime_priority = bool(realtime_priority)
        self.thread_priority: str | None = None  # what the loop thread got
        self._write_grace_until = 0.0  # perf_counter_ns until which a write may delay telemetry
        self.device_name = device_name
        self._transport_kind = transport
        # Firmware joint feed-forward (constant, velocity table and position
        # table); None leaves the hand's current setting untouched.
        self.feed_forward = None if feed_forward is None else bool(feed_forward)

        self.hand = None
        self.transport = None
        self.state: dict | None = None
        self.initial_qpos: np.ndarray | None = None
        self.q_desired = self.qd_desired = self.effort_ff = self.torque = None
        self.error: str | None = None
        self.warnings: set[str] = set()
        self.stats = {"ticks": 0, "sends": 0, "overruns": 0, "max_interval_ms": 0.0,
                      "max_tracking_error": 0.0}

        self._mode = "impedance"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._manager = None
        self._describe = None
        self._status = None  # (host_ns, [JointStatus])
        self._motion = None  # (start_ns, ruckig trajectory, goal, done event)
        self._recording: _Recording | None = None
        self._saved_gains = None
        self._switch_interval: float | None = None
        self._enable_requested = self._enabled = self._shut_down = False
        self._update_freq = 50.0
        self._last_update_time: dict[str, float] = {}

    @property
    def dof(self) -> int:
        return DOF

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def kp(self) -> np.ndarray:
        return self._kp.copy()

    @kp.setter
    def kp(self, value) -> None:
        self.set_gains(kp=value)

    @property
    def kd(self) -> np.ndarray:
        return self._kd.copy()

    @kd.setter
    def kd(self, value) -> None:
        self.set_gains(kd=value)

    def set_gains(self, kp=None, kd=None) -> None:
        """Set kp and/or kd (scalar or per joint) with a single write to the hand."""
        new_kp = self._kp if kp is None else _as_vector(kp, "kp", nonnegative=True)
        new_kd = self._kd if kd is None else _as_vector(kd, "kd", nonnegative=True)
        self._kp, self._kd = new_kp, new_kd
        self._sync_gains()

    @property
    def friction(self) -> np.ndarray:
        """Per-joint Coulomb friction feedforward (A) for impedance mode.

        While the target moves, `friction * tanh(qd_desired / friction_velocity)`
        is added to the effort, so the PD only supplies the difference instead
        of winding up an error until the joint breaks free (stick-slip).
        """
        return self._friction.copy()

    @friction.setter
    def friction(self, value) -> None:
        friction = _as_vector(value, "friction", nonnegative=True)
        if np.any(friction > self.effort_limit):
            raise ValueError(f"friction exceeds effort_limit {self.effort_limit} A")
        self._friction = friction

    # ---- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        # CPython hands the GIL between busy threads every 5 ms by default,
        # which stalls the 1 kHz loop; shorten it while the controller runs.
        self._switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(min(self._switch_interval, SWITCH_INTERVAL_S))
        try:
            await asyncio.to_thread(self._prepare)
            self._thread = threading.Thread(target=self._run, name="aiowuji-loop", daemon=True)
            self._thread.start()
            if not self.dry_run:
                await asyncio.to_thread(self._enable)
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        await asyncio.to_thread(self._shutdown)

    def _prepare(self) -> None:
        sdk = import_sdk()
        self._describe = sdk.WujiHand2.describe_error
        if self._hand_arg is None or isinstance(self._hand_arg, str):
            self._manager, self.hand = connect(self._hand_arg, device_name=self.device_name)
        else:
            self.hand = self._hand_arg  # the caller owns this connection
        online = self.hand.online_joints_count().get()
        if online != DOF:
            raise RuntimeError(f"expected {DOF} online joints, got {online}")
        self.transport = open_transport(sdk, self.hand, publish=not self.dry_run,
                                         kind=self._transport_kind)

        deadline, state, status = time.monotonic() + 2.0, None, None
        while (state is None or status is None) and time.monotonic() < deadline:
            state = self.transport.latest_state() or state
            status = self.transport.latest_status() or status
            time.sleep(0.002)
        if state is None or status is None:
            raise RuntimeError("no joint_states/joint_diagnostics within 2 s")
        if None in state.qpos:
            raise RuntimeError("joint_states is missing joints")
        not_ready = [s.nid for s in status if s.ext_state != _READY]
        faults = self._faults(status)
        if not_ready or faults or len(status) != DOF:
            raise RuntimeError(f"refusing to start: not Ready {not_ready}, faults {faults}")
        if outside_limits(state.qpos):
            raise RuntimeError(f"joints {outside_limits(state.qpos)} are outside their documented range")
        now = time.perf_counter_ns()
        self._set_state(state)
        self._status = (now, status)
        self.initial_qpos = self.state["qpos"].copy()
        self.q_desired = self.initial_qpos.copy()
        self.qd_desired, self.effort_ff, self.torque = np.zeros(DOF), np.zeros(DOF), np.zeros(DOF)
        if self.dry_run:
            return

        if self.feed_forward is not None:
            self._apply_feed_forward(sdk, self.feed_forward)
        saved = (self.hand.effort_limit().get(), self.hand.mit_params().get())
        if None in saved[0] or None in saved[1]:
            raise RuntimeError("cannot read the current gains of every joint")
        self._saved_gains = saved
        self.hand.effort_limit().set(self.effort_limit)
        gains = (self._kp, self._kd) if self._mode == "impedance" else (np.zeros(DOF),) * 2
        self._push_gains(*gains)
        limits, readback = self.hand.effort_limit().get(), self.hand.mit_params().get()
        if not (np.allclose(limits, self.effort_limit, rtol=0.0, atol=1e-5)
                and np.allclose([g.kp for g in readback], gains[0], rtol=0.0, atol=1e-5)
                and np.allclose([g.kd for g in readback], gains[1], rtol=0.0, atol=1e-5)):
            raise RuntimeError("gain readback does not match the requested values")

    def _apply_feed_forward(self, sdk, on: bool) -> None:
        # wuji-sdk 2026.9.22 with firmware 2.10.0+. The firmware keeps it in RAM
        # and only accepts writes, so it is set before enabling and not restored.
        if not hasattr(self.hand, "set_constant"):
            raise RuntimeError("feed_forward requires wuji-sdk 2026.9.22 or later")
        mode = sdk.FeedForwardMode.ConstantAndVelocityTable if on else sdk.FeedForwardMode.Off
        self.hand.set_constant(mode)
        self.hand.set_position_table(on)

    def _enable(self) -> None:
        # The loop already streams the measured pose, so the firmware target
        # equals the current position when the motors engage.
        time.sleep(0.02)
        requested = time.perf_counter_ns()
        self._enable_requested = True
        with self._writing():
            self.hand.enable()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and self.error is None:
            received, status = self._status
            if received > requested and all(s.ext_state == _ENABLED for s in status):
                self._enabled = True
                return
            time.sleep(0.005)
        raise RuntimeError(self.error or "not every joint reached Enabled within 2 s")

    def _shutdown(self) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._switch_interval is not None:
            sys.setswitchinterval(self._switch_interval)
        problems = []
        disabled = not self._enable_requested
        if self._enable_requested:
            disabled = self._disable_and_confirm(self.hand.disable)
            if not disabled:
                problems.append("disable not confirmed; sent emergency stop")
                disabled = self._disable_and_confirm(self.hand.emergency_stop)
        if self.transport is not None:
            problems += self.transport.close()
        if self._saved_gains is not None:
            if disabled:
                try:
                    self.hand.effort_limit().set(list(self._saved_gains[0]))
                    self.hand.mit_params().set([(g.kp, g.kd) for g in self._saved_gains[1]])
                except Exception as error:
                    problems.append(f"restoring gains failed: {error}")
            else:
                problems.append("gains not restored because disable was not confirmed")
        if self._manager is not None:
            try:
                self._manager.disconnect(self.device_name)
            except Exception as error:
                problems.append(f"disconnect failed: {error}")
        if not disabled:
            raise RuntimeError("; ".join(problems))
        if problems and self.error is None:
            self.error = "; ".join(problems)

    def _disable_and_confirm(self, action, timeout_s: float = 1.0) -> bool:
        """Run disable/emergency stop, then require a fresh frame with no joint Enabled."""
        try:
            action()
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                status = self.transport.latest_status()
                if status is not None and not any(s.ext_state == _ENABLED for s in status):
                    return True
                time.sleep(0.005)
        except Exception:
            pass
        return False

    # ---- commands --------------------------------------------------------

    def switch(self, controller_type: str) -> None:
        if controller_type not in ("impedance", "torque"):
            raise ValueError(f"Unknown controller type: {controller_type}")
        if self.state is None:
            self._mode = controller_type
            return
        self._raise_if_stopped()
        q = self.state["qpos"].copy()
        with self._lock:
            self._cancel_motion()
            self.q_desired, self.qd_desired = q, np.zeros(DOF)
            self.effort_ff, self.torque = np.zeros(DOF), np.zeros(DOF)
            # Torque mode streams the measured pose with zero effort, so the
            # firmware target is current while its gains change in either direction.
            self._mode = "torque"
        if self._saved_gains is not None:
            gains = (self._kp, self._kd) if controller_type == "impedance" else (np.zeros(DOF),) * 2
            self._push_gains(*gains)
        with self._lock:
            self._mode = controller_type
        self._last_update_time.clear()

    def set_freq(self, freq: float) -> None:
        if freq <= 0:
            raise ValueError("freq must be positive")
        self._update_freq = float(freq)

    async def set(self, attr: str, value) -> None:
        """Set a target, pacing repeated calls to `set_freq()` Hz."""
        now = time.perf_counter()
        target_time = self._last_update_time.get(attr, now) + 1.0 / self._update_freq
        self.set_now(attr, value)
        self._last_update_time[attr] = max(target_time, now)
        if target_time > now:
            await asyncio.sleep(target_time - now)

    def set_now(self, attr: str, value) -> None:
        if self.state is None:
            raise RuntimeError("Call await controller.start() first")
        self._raise_if_stopped()
        if attr not in _TARGETS:
            raise ValueError(f"attr must be one of {_TARGETS}")
        arr = _as_vector(value, attr)
        if attr == "q_desired" and outside_limits(arr):
            raise ValueError(f"q_desired joints {outside_limits(arr)} are outside their documented range")
        if attr in ("effort_ff", "torque") and np.any(np.abs(arr) > self.effort_limit):
            raise ValueError(f"{attr} exceeds effort_limit {self.effort_limit} A")
        with self._lock:
            if attr != "torque":
                self._cancel_motion()
            setattr(self, attr, arr)

    async def move(self, qpos=None, *, max_velocity=1.0, max_acceleration=4.0, max_jerk=40.0) -> None:
        """Move to a joint target (default: all zeros, the vendor home pose).

        Ruckig generates a jerk-limited trajectory from the current target; the
        loop samples its position and velocity at `rate_hz` as impedance targets.
        """
        if self.state is None:
            raise RuntimeError("Call await controller.start() before move()")
        self._raise_if_stopped()
        from ruckig import InputParameter, Result, Ruckig, Trajectory

        if self._mode != "impedance":
            self.switch("impedance")
        goal = np.zeros(DOF) if qpos is None else _as_vector(qpos, "qpos")
        if outside_limits(goal):
            raise ValueError(f"qpos joints {outside_limits(goal)} are outside their documented range")
        with self._lock:
            start, start_vel = self.q_desired.copy(), self.qd_desired.copy()

        inp = InputParameter(DOF)
        inp.current_position = start.tolist()
        inp.current_velocity = start_vel.tolist()
        inp.current_acceleration = [0.0] * DOF
        inp.target_position = goal.tolist()
        inp.target_velocity = [0.0] * DOF
        inp.target_acceleration = [0.0] * DOF
        inp.max_velocity = _as_vector(max_velocity, "max_velocity").tolist()
        inp.max_acceleration = _as_vector(max_acceleration, "max_acceleration").tolist()
        inp.max_jerk = _as_vector(max_jerk, "max_jerk").tolist()
        trajectory = Trajectory(DOF)
        result = Ruckig(DOF).calculate(inp, trajectory)
        if result not in (Result.Working, Result.Finished):
            raise RuntimeError(f"Ruckig failed to generate trajectory: {result}")

        done = threading.Event()
        with self._lock:
            self._cancel_motion()
            self._motion = (time.perf_counter_ns(), trajectory, goal, done)
        while not done.is_set():
            self._raise_if_stopped()
            await asyncio.sleep(0.005)
        self._raise_if_stopped()

    async def wait(self, seconds: float) -> None:
        """Sleep while the loop holds the current target; raise if it stopped."""
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0:
            self._raise_if_stopped()
            await asyncio.sleep(min(0.01, remaining))
        self._raise_if_stopped()

    def start_recording(self, seconds: float = 60.0) -> None:
        """Record every loop tick (schedule, command sent, newest telemetry)
        until `stop_recording()` or `seconds` of ticks have been stored."""
        self._recording = _Recording(int(seconds * self.rate_hz) + 1)

    def stop_recording(self) -> dict[str, np.ndarray]:
        """Stop recording and return the per-tick arrays (times in ns, angles in rad)."""
        recording, self._recording = self._recording, None
        if recording is None:
            raise RuntimeError("start_recording() was not called")
        return recording.arrays()

    # ---- internals -------------------------------------------------------

    def _raise_if_stopped(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"controller stopped: {self.error}")
        if self._shut_down:
            raise RuntimeError("controller is stopped")

    def _cancel_motion(self) -> None:
        if self._motion is not None:
            self._motion[3].set()
            self._motion = None

    @contextlib.contextmanager
    def _writing(self):
        """Mark a write to the hand, which delays its joint_states while it runs."""
        self._write_grace_until = float("inf")
        try:
            yield
        finally:
            self._write_grace_until = time.perf_counter_ns() + WRITE_GRACE_S * 1e9

    def _push_gains(self, kp, kd) -> None:
        with self._writing():
            self.hand.mit_params().set([(float(p), float(d)) for p, d in zip(kp, kd)])

    def _sync_gains(self) -> None:
        if self._saved_gains is not None and self._mode == "impedance" and not self._shut_down:
            self._push_gains(self._kp, self._kd)

    def _set_state(self, state) -> None:
        self.state = {"qpos": np.array(state.qpos), "qvel": np.array(state.qvel),
                      "effort": np.array(state.effort), "seq": state.seq,
                      "timestamp_us": state.timestamp_us, "host_ns": time.perf_counter_ns()}

    def _faults(self, status) -> list[str]:
        faults = []
        for joint in status:
            if joint.error_code:
                info = self._describe(joint.error_code)
                if info is not None and info["severity"] == "Warning":
                    self.warnings.add(info["name"])
                else:
                    name = info["name"] if info is not None else "unknown"
                    faults.append(f"NID {joint.nid}: error 0x{joint.error_code:04x} ({name})")
            if joint.ext_state == _STOPPED or (self._enabled and joint.ext_state != _ENABLED):
                faults.append(f"NID {joint.nid}: ext_state {joint.ext_state}")
        if self._enabled and len(status) != DOF:
            faults.append(f"{DOF - len(status)} joints missing from diagnostics")
        return faults

    def _fail(self, reason: str) -> None:
        if self.error is None:
            self.error = reason
        if self._enable_requested:
            try:
                self.hand.disable()
            except Exception:
                pass  # stop() confirms the disable and escalates to emergency stop

    def _run(self) -> None:
        period = round(1e9 / self.rate_hz)
        stale_ns, write_stale_ns = self.stale_ms * 1e6, WRITE_STALE_MS * 1e6
        if self.realtime_priority:
            self.thread_priority = _prioritize_current_thread(1.0 / self.rate_hz)
        deadline = previous = time.perf_counter_ns()
        last_state = self.state["host_ns"]
        tick = 0
        try:
            while not self._stop.is_set():
                now = time.perf_counter_ns()
                if deadline > now:
                    time.sleep((deadline - now) / 1e9)
                    now = time.perf_counter_ns()
                if now - deadline >= period:
                    self.stats["overruns"] += 1
                    deadline = now  # shift the schedule; never send a catch-up burst
                scheduled = deadline
                deadline += period
                self.stats["max_interval_ms"] = max(self.stats["max_interval_ms"], (now - previous) / 1e6)
                previous = now

                state = self.transport.latest_state()
                if state is not None:
                    if None in state.qpos:
                        return self._fail("a joint is missing from joint_states")
                    self._set_state(state)
                    last_state = now
                if now - last_state > (write_stale_ns if now < self._write_grace_until else stale_ns):
                    return self._fail(f"no joint_states for {(now - last_state) / 1e6:.1f} ms")
                if tick % self.transport.status_every == 0:
                    status = self.transport.latest_status()
                    if status is not None:
                        self._status = (now, status)
                    faults = self._faults(self._status[1])
                    if faults:
                        return self._fail(f"joint fault: {faults}")

                q = self.state["qpos"]
                with self._lock:
                    if self._mode == "impedance":
                        if self._motion is not None:
                            self._step_motion(now)
                        effort = self.effort_ff
                        if self._friction.any():
                            effort = np.clip(
                                effort + self._friction * np.tanh(self.qd_desired / self.friction_velocity),
                                -self.effort_limit, self.effort_limit)
                        command = (self.q_desired, self.qd_desired, effort)
                    else:
                        command = (q, np.zeros(DOF), self.torque)
                    tracking = self._mode == "impedance" and self._enabled
                if tracking:
                    error = float(np.max(np.abs(command[0] - q)))
                    self.stats["max_tracking_error"] = max(self.stats["max_tracking_error"], error)
                    if self.max_tracking_error is not None and error > self.max_tracking_error:
                        return self._fail(f"tracking error {error:.3f} rad exceeds {self.max_tracking_error:.3f}")
                send_started = time.perf_counter_ns()
                if self.transport.publisher is not None:
                    self.transport.send(*command)
                    self.stats["sends"] += 1
                recording = self._recording
                if recording is not None:
                    recording.append(now, scheduled, time.perf_counter_ns() - send_started,
                                     command, self.state)
                tick += 1
                self.stats["ticks"] = tick
        except Exception as error:
            self._fail(f"control loop error: {error}")

    def _step_motion(self, now: int) -> None:
        started, trajectory, goal, done = self._motion
        t = (now - started) / 1e9
        if t >= trajectory.duration:
            self.q_desired, self.qd_desired = goal.copy(), np.zeros(DOF)
            self._motion = None
            done.set()
        else:
            position, velocity, _ = trajectory.at_time(t)
            self.q_desired, self.qd_desired = np.asarray(position), np.asarray(velocity)
