"""Kinematic clearance checks with the official Wuji Hand 2 MuJoCo model.

Only kinematics and collision geometry are used: set the 20 joint angles and
measure how close one finger's collision meshes (their convex hulls, geom
group 2) come to the rest of the hand. Needs `mujoco`
(`pip install "aiowuji[sim]"`); the model is bundled in `aiowuji/models`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .hand import DOF, FINGERS

MODEL_DIR = Path(__file__).resolve().parent / "models" / "wuji_hand2"
_JOINTS = {"thumb": ("thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip"),
           "pinky": ("pinky_mcp_flex", "pinky_mcp_abd", "pinky_pip", "pinky_dip")}
for _finger in ("index", "middle", "ring"):
    _JOINTS[_finger] = tuple(f"{_finger}_finger_{j}" for j in ("mcp_flex", "mcp_abd", "pip", "dip"))


class HandModel:
    """The Wuji Hand 2 (Beta 2) model, posed with aiowuji's 20 joint angles (rad)."""

    def __init__(self, handedness: str = "right") -> None:
        import mujoco

        self._mujoco = mujoco
        side = {"right": "r_", "left": "l_"}[handedness]
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "mjcf" / f"{handedness}.xml"))
        self.data = mujoco.MjData(self.model)
        names = [side + name for finger in FINGERS for name in _JOINTS[finger]]
        self._qadr = np.array([self.model.jnt_qposadr[self.model.joint(name).id] for name in names])
        wrist = self.model.body(f"{side}wrist").id
        self.geoms = {part: [] for part in (*FINGERS, "palm")}
        for geom in np.flatnonzero(self.model.geom_group == 2):
            body = self.model.geom_bodyid[geom]
            while self.model.body_parentid[body] not in (wrist, 0):
                body = self.model.body_parentid[body]
            name = self.model.body(body).name
            part = "palm" if body in (wrist, 0) else next(f for f in FINGERS if name.startswith(side + f))
            self.geoms[part].append(int(geom))
        self._fromto = np.zeros(6)

    def pose(self, q) -> None:
        q = np.asarray(q, float)
        if q.shape != (DOF,):
            raise ValueError(f"expected {DOF} joint angles")
        self.data.qpos[self._qadr] = q
        self._mujoco.mj_kinematics(self.model, self.data)

    def distance(self, a: int, b: int, cutoff: float) -> float:
        """Closest distance between two posed geoms (m), capped at `cutoff`."""
        gap = np.linalg.norm(self.data.geom_xpos[a] - self.data.geom_xpos[b])
        if gap - self.model.geom_rbound[a] - self.model.geom_rbound[b] > cutoff:
            return cutoff  # bounding spheres already farther apart
        return self._mujoco.mj_geomDistance(self.model, self.data, a, b, cutoff, self._fromto)

    def clearance(self, finger: str, reference, margin: float) -> "Clearance":
        return Clearance(self, finger, reference, margin)


class Clearance:
    """How far a finger stays from the rest of the hand, relative to a reference pose.

    Every pair of (finger geom, other geom) must stay at least `margin` apart.
    Pairs of fingers already closer than that in the reference pose (the
    knuckle bases sit 2-5 mm apart) may close to half their reference gap but
    never touch. Pairs with the palm closer than `margin` in the reference pose
    are the finger's own joint (the palm's convex hull wraps the knuckles) and
    are skipped; farther ones, such as a fingertip curling in, are checked.
    """

    def __init__(self, model: HandModel, finger: str, reference, margin: float) -> None:
        self.model, self.margin = model, float(margin)
        self.cutoff = self.margin + 0.01
        model.pose(reference)
        self.pairs, self.floors = [], []
        for part, geoms in model.geoms.items():
            if part == finger:
                continue
            for a in model.geoms[finger]:
                for b in geoms:
                    gap = model.distance(a, b, self.cutoff)
                    if part == "palm" and gap < self.margin:
                        continue
                    self.pairs.append((a, b))
                    self.floors.append(min(self.margin, 0.5 * gap))

    def slack(self, q) -> float:
        """Smallest (distance - allowed minimum) over the pairs, in m; negative is too close."""
        self.model.pose(q)
        return min(self.model.distance(a, b, self.cutoff) - floor for (a, b), floor in zip(self.pairs, self.floors))
