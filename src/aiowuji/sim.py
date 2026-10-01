"""Simulate the Wuji Hand 2 with the firmware's joint PD running on the host.

`scripts/sysid_fit.py` writes a MuJoCo model whose actuators are plain motors:
ctrl is motor current in A (the SDK's effort unit) and each actuator's gear is
that joint's torque constant in N m/A. The PD is not in the model.
`firmware_effort` reproduces the law the hand runs, as measured in the sysid
recordings:

    effort = clip(kp_scale * kp * (q_desired - q) + kd_scale * kd * (qd_desired - qd) + effort_ff,
                  -effort_limit, effort_limit)

with kp (A/rad), kd (A s/rad) and effort_limit (A) exactly as sent to the hand.
The documented law has no scale; the hand's measured effort is about 1.95 times
it. The fitted model stores kp_scale, kd_scale and the command lag (physics
steps from sending a target to the hand using it) as custom numerics.

`WujiSim` runs the PD at every physics step (1 kHz, like the firmware) and
takes a new target every `decimation` steps (20 for 50 Hz control). With many
simulations, e.g. an mjbatch `Batch`, call `firmware_effort` on the bound
qpos/qvel arrays and write the result to ctrl before each step; it broadcasts.
Needs `mujoco` (`pip install "aiowuji[sim]"`).
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from .hand import DOF, FINGERS
from .model import _JOINTS

# Custom numerics the fitted model carries, with the values used when absent.
FIRMWARE_NUMERICS = {"aiowuji_kp_scale": 1.95, "aiowuji_kd_scale": 1.9, "aiowuji_lag_steps": 2.0}


def mjcf_joints(handedness: str = "right") -> list[str]:
    """The model's joint names in aiowuji's joint order."""
    side = {"right": "r_", "left": "l_"}[handedness]
    return [side + name for finger in FINGERS for name in _JOINTS[finger]]


def firmware_effort(q_desired, qd_desired, q, qd, kp, kd, effort_limit, kp_scale, kd_scale, effort_ff=0.0):
    """The motor current (A) the hand's joint PD commands; every argument broadcasts."""
    effort = kp_scale * kp * (q_desired - q) + kd_scale * kd * (qd_desired - qd) + effort_ff
    return np.clip(effort, -effort_limit, effort_limit)


def firmware_constants(model) -> dict[str, float]:
    """kp_scale, kd_scale and lag_steps stored in a fitted model (defaults if missing)."""
    out = {}
    for name, default in FIRMWARE_NUMERICS.items():
        try:
            out[name.removeprefix("aiowuji_")] = float(model.numeric(name).data[0])
        except KeyError:
            out[name.removeprefix("aiowuji_")] = default
    return out


class WujiSim:
    """One simulated hand: set targets at `control_hz`, the PD runs every physics step.

    `step(q_desired)` advances `decimation` physics steps. The PD switches to the
    new target `lag_steps` physics steps after it is sent and tracks the previous
    one until then, as the hand does while a command is in flight. Joint arrays
    are in aiowuji order.
    """

    def __init__(self, xml_path, *, kp=3.0, kd=0.05, effort_limit: float = 0.5,
                 control_hz: float = 50.0, lag_steps: int | None = None) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(xml_path)))
        self.data = mujoco.MjData(self.model)
        if self.model.nu != DOF:
            raise ValueError(f"expected {DOF} actuators, got {self.model.nu}")
        constants = firmware_constants(self.model)
        self.kp_scale, self.kd_scale = constants["kp_scale"], constants["kd_scale"]
        self.lag_steps = int(round(constants["lag_steps"])) if lag_steps is None else int(lag_steps)
        steps = 1.0 / (control_hz * self.model.opt.timestep)
        self.decimation = int(round(steps))
        if abs(steps - self.decimation) > 1e-6 or self.decimation < 1 or self.lag_steps < 0:
            raise ValueError(f"control_hz {control_hz} is not a whole number of physics steps")
        joints = self.model.actuator_trnid[:, 0]
        self._qadr, self._dadr = self.model.jnt_qposadr[joints], self.model.jnt_dofadr[joints]
        self.kp, self.kd = np.broadcast_to(kp, DOF).astype(float), np.broadcast_to(kd, DOF).astype(float)
        self.effort_limit = float(effort_limit)
        self.effort = np.zeros(DOF)
        self.reset()

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self._qadr].copy()

    @property
    def qd(self) -> np.ndarray:
        return self.data.qvel[self._dadr].copy()

    def reset(self, qpos=None) -> None:
        """Reset to `qpos` (default zeros) at rest, holding it as the target."""
        self._mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            self.data.qpos[self._qadr] = qpos
        self._mujoco.mj_forward(self.model, self.data)
        self._target = (self.q, np.zeros(DOF), np.zeros(DOF))
        self._pending = deque()  # (physics step it takes effect, target)
        self._steps = 0

    def step(self, q_desired, qd_desired=None, effort_ff=None) -> None:
        """Send a new target and advance one control period."""
        self._pending.append((self._steps + self.lag_steps, (
            np.asarray(q_desired, float),
            np.zeros(DOF) if qd_desired is None else np.asarray(qd_desired, float),
            np.zeros(DOF) if effort_ff is None else np.asarray(effort_ff, float))))
        for _ in range(self.decimation):
            while self._pending and self._pending[0][0] <= self._steps:
                self._target = self._pending.popleft()[1]
            q_des, qd_des, ff = self._target
            self.effort = firmware_effort(q_des, qd_des, self.data.qpos[self._qadr], self.data.qvel[self._dadr],
                                          self.kp, self.kd, self.effort_limit, self.kp_scale, self.kd_scale, ff)
            self.data.ctrl[:] = self.effort
            self._mujoco.mj_step(self.model, self.data)
            self._steps += 1
