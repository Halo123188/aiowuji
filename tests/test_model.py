"""The bundled Wuji Hand 2 model and its clearance checks (needs mujoco)."""
from __future__ import annotations

import unittest

import numpy as np

try:
    import mujoco  # noqa: F401
except ImportError:
    mujoco = None

from aiowuji import JOINT_LIMITS


@unittest.skipIf(mujoco is None, "mujoco not installed")
class HandModelTests(unittest.TestCase):
    def setUp(self) -> None:
        from aiowuji.model import HandModel
        self.model = HandModel("right")

    def test_joint_ranges_match_the_documented_limits(self) -> None:
        m = self.model.model
        for index, (low, high) in enumerate(JOINT_LIMITS):
            joint = int(np.flatnonzero(m.jnt_qposadr == self.model._qadr[index])[0])
            np.testing.assert_allclose(m.jnt_range[joint], (low, high), atol=2e-3)

    def test_home_is_clear_and_spreading_into_a_neighbour_is_not(self) -> None:
        home = np.zeros(20)
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            self.assertGreaterEqual(self.model.clearance(finger, home, 0.005).slack(home), 0.0)
        check = self.model.clearance("index", home, 0.005)
        toward_middle, toward_thumb = home.copy(), home.copy()
        toward_middle[5], toward_thumb[5] = 0.3, -0.3  # + spreads toward the pinky side
        self.assertLess(check.slack(toward_middle), 0.0)
        self.assertGreaterEqual(check.slack(toward_thumb), 0.0)

    def test_the_finger_can_flex_fully_on_its_own(self) -> None:
        home = np.zeros(20)
        check = self.model.clearance("middle", home, 0.005)
        for angle in (-1.0, 1.5):
            q = home.copy()
            q[8] = angle
            self.assertGreaterEqual(check.slack(q), 0.0)


if __name__ == "__main__":
    unittest.main()
