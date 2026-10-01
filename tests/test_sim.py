"""The host-side firmware PD and WujiSim's control loop (needs mujoco)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import mujoco  # noqa: F401
except ImportError:
    mujoco = None

from aiowuji.sim import firmware_effort

# 20 unit-inertia hinges with plain motors, no gravity, and the firmware numerics.
_JOINTS = "".join(
    f'<body name="b{i}"><joint name="j{i}" axis="1 0 0" armature="1"/>'
    f'<geom size="0.01" mass="1e-6"/></body>' for i in range(20))
_MOTORS = "".join(f'<motor joint="j{i}" ctrlrange="-2 2"/>' for i in range(20))
_XML = f"""<mujoco><option timestep="0.001" gravity="0 0 0"/>
  <custom><numeric name="aiowuji_kp_scale" data="2"/><numeric name="aiowuji_kd_scale" data="1"/>
  <numeric name="aiowuji_lag_steps" data="2"/></custom>
  <worldbody>{_JOINTS}</worldbody><actuator>{_MOTORS}</actuator></mujoco>"""


class FirmwareEffortTests(unittest.TestCase):
    def test_scales_the_documented_law_and_clips_at_the_limit(self) -> None:
        effort = firmware_effort(np.array([0.05, 1.0]), 0.0, 0.0, np.array([0.5, 0.0]), kp=3.0, kd=0.1,
                                 effort_limit=0.5, kp_scale=2.0, kd_scale=1.5)
        np.testing.assert_allclose(effort, [2.0 * 3.0 * 0.05 - 1.5 * 0.1 * 0.5, 0.5])


@unittest.skipIf(mujoco is None, "mujoco not installed")
class WujiSimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.xml = Path(self.dir.name) / "hand.xml"
        self.xml.write_text(_XML)

    def tearDown(self) -> None:
        self.dir.cleanup()

    def test_reads_the_firmware_constants_and_decimates_to_the_control_rate(self) -> None:
        from aiowuji.sim import WujiSim

        sim = WujiSim(self.xml, control_hz=50)
        self.assertEqual((sim.kp_scale, sim.kd_scale, sim.lag_steps, sim.decimation), (2.0, 1.0, 2, 20))
        sim.step(np.full(20, 0.1))
        self.assertAlmostEqual(sim.data.time, 0.02)
        self.assertTrue(np.all(sim.q > 0))

    def test_a_target_takes_effect_lag_steps_after_it_is_sent(self) -> None:
        from aiowuji.sim import WujiSim

        sim = WujiSim(self.xml, kp=1.0, kd=0.0, control_hz=1000)
        efforts = []
        for _ in range(4):
            sim.step(np.full(20, 0.1))
            efforts.append(sim.effort[0])
        np.testing.assert_allclose(efforts[:2], 0.0)
        self.assertAlmostEqual(efforts[2], 2.0 * 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
