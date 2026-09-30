"""WujiController against a fake hand that speaks the public wuji_sdk API.

The fake exercises the same SdkTransport path used with the official Linux SDK.
It checks controller logic only; it says nothing about firmware behavior.
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from aiowuji import WujiController
from aiowuji.hand import JOINT_NIDS


class FakeSubscription:
    def __init__(self, make_frame) -> None:
        self.make_frame = make_frame
        self.fresh = False
        self.closed = False

    def recv(self):
        # One new frame per drain, like a 1 kHz stream polled at 1 kHz.
        self.fresh = not self.fresh
        return self.make_frame() if self.fresh else None

    def close(self) -> None:
        self.closed = True


class FakePublisher:
    def __init__(self, hand) -> None:
        self.hand = hand
        self.closed = False

    def send(self, joints) -> None:
        self.hand.sent.append([(j.position, j.velocity, j.effort) for j in joints])
        if self.hand.follow:
            self.hand.q = [j.position for j in joints]

    def close(self) -> None:
        self.closed = True


class FakeHand:
    serial_number = "FAKE_HAND"

    def __init__(self) -> None:
        self.q = [0.1] * 20
        self.ext_state, self.error_code = 1, 0
        self.streaming, self.follow = True, True
        self.gains = [SimpleNamespace(kp=1.5, kd=0.1)] * 20
        self.limits = [1.5] * 20
        self.calls, self.sent, self.subscriptions = [], [], []
        self.seq = 0

    def _state(self):
        if not self.streaming:
            return None
        self.seq += 1
        joints = [SimpleNamespace(nid=nid, position=self.q[j], velocity=0.0, effort=0.0)
                  for j, nid in enumerate(JOINT_NIDS)]
        return SimpleNamespace(header=SimpleNamespace(seq=self.seq, timestamp_us=self.seq), joints=joints)

    def _diagnostics(self):
        return SimpleNamespace(joints=[
            SimpleNamespace(nid=nid, status_word=SimpleNamespace(ext_state=self.ext_state),
                            error_code_current=self.error_code) for nid in JOINT_NIDS])

    def _subscribe(self, make_frame):
        subscription = FakeSubscription(make_frame)
        self.subscriptions.append(subscription)
        return subscription

    def joint_states(self):
        return SimpleNamespace(subscribe=lambda: self._subscribe(self._state))

    def joint_diagnostics(self):
        return SimpleNamespace(subscribe=lambda: self._subscribe(self._diagnostics))

    def joint_command(self):
        return SimpleNamespace(publish=lambda: FakePublisher(self))

    def online_joints_count(self):
        return SimpleNamespace(get=lambda: 20)

    def mit_params(self):
        def set_gains(pairs):
            self.calls.append(("mit_params", pairs[0]))
            self.gains = [SimpleNamespace(kp=kp, kd=kd) for kp, kd in pairs]
        return SimpleNamespace(get=lambda: list(self.gains), set=set_gains)

    def effort_limit(self):
        def set_limit(value):
            self.calls.append(("effort_limit", value))
            self.limits = list(value) if isinstance(value, list) else [value] * 20
        return SimpleNamespace(get=lambda: list(self.limits), set=set_limit)

    def set_constant(self, mode) -> None:
        self.calls.append(("set_constant", int(mode)))

    def set_position_table(self, on) -> None:
        self.calls.append(("set_position_table", bool(on)))

    def enable(self) -> None:
        self.calls.append("enable")
        self.ext_state = 2

    def disable(self) -> None:
        self.calls.append("disable")
        self.ext_state = 1

    def emergency_stop(self) -> None:
        self.calls.append("emergency_stop")
        self.ext_state = 3


def run(coroutine):
    return asyncio.run(asyncio.wait_for(coroutine, 10))


class ControllerTests(unittest.TestCase):
    def test_lifecycle_moves_switches_and_restores_gains(self) -> None:
        hand = FakeHand()

        async def scenario():
            controller = WujiController(hand, transport="sdk")
            await controller.start()
            self.assertEqual(hand.calls, [("effort_limit", 0.5), ("mit_params", (3.0, 0.05)), "enable"])
            target = controller.initial_qpos.copy()
            target[0] += 0.05
            await controller.move(target)
            self.assertAlmostEqual(hand.sent[-1][0][0], target[0], places=6)
            self.assertEqual(hand.sent[-1][0][1], 0.0)

            controller.switch("torque")
            self.assertEqual(hand.calls[-1], ("mit_params", (0.0, 0.0)))
            await controller.set("torque", np.full(20, 0.1))
            await controller.wait(0.02)
            self.assertAlmostEqual(hand.sent[-1][5][2], 0.1)

            controller.switch("impedance")
            self.assertEqual(hand.calls[-1], ("mit_params", (3.0, 0.05)))
            controller.kp = 4.0
            self.assertEqual(hand.calls[-1], ("mit_params", (4.0, 0.05)))
            await controller.stop()
            return controller

        controller = run(scenario())
        self.assertIsNone(controller.error)
        self.assertEqual(hand.calls[-3:], ["disable", ("effort_limit", [1.5] * 20), ("mit_params", (1.5, 0.1))])
        self.assertTrue(all(s.closed for s in hand.subscriptions))
        self.assertGreater(controller.stats["sends"], 20)

    def test_recording_captures_every_tick_of_a_move(self) -> None:
        hand = FakeHand()

        async def scenario():
            controller = WujiController(hand, transport="sdk")
            await controller.start()
            controller.start_recording(seconds=5)
            target = controller.initial_qpos.copy()
            target[16] += 0.05
            await controller.move(target)
            ticks = controller.stop_recording()
            await controller.stop()
            return controller, ticks

        controller, ticks = run(scenario())
        if sys.platform == "darwin":
            self.assertEqual(controller.thread_priority, "qos_user_interactive")
        n = len(ticks["tick_ns"])
        self.assertGreater(n, 50)
        self.assertTrue(all(len(values) == n for values in ticks.values()))
        self.assertTrue(np.all(np.diff(ticks["tick_ns"]) > 0))
        self.assertTrue(np.all(ticks["scheduled_ns"] <= ticks["tick_ns"]))
        self.assertAlmostEqual(float(ticks["q_cmd"][-1, 16]), 0.15, places=5)
        self.assertEqual(ticks["q"].shape, (n, 20))

    def test_friction_feedforward_follows_the_moving_target_only(self) -> None:
        hand = FakeHand()
        with self.assertRaisesRegex(ValueError, "effort_limit"):
            WujiController(hand, friction=0.6)

        async def scenario():
            controller = WujiController(hand, transport="sdk", friction=0.1)
            await controller.start()
            controller.start_recording(seconds=5)
            target = controller.initial_qpos.copy()
            target[16] -= 0.05
            await controller.move(target)
            await controller.wait(0.02)
            ticks = controller.stop_recording()
            await controller.stop()
            return ticks

        ticks = run(scenario())
        effort = ticks["effort_cmd"]
        self.assertLess(effort[:, 16].min(), -0.09)          # pushes along the (negative) motion
        self.assertLessEqual(effort[:, 16].max(), 1e-6)
        self.assertEqual(float(effort[-1, 16]), 0.0)         # nothing once the target stops
        self.assertEqual(float(np.abs(np.delete(effort, 16, axis=1)).max()), 0.0)

    def test_tracking_error_disables_and_stops_waiters(self) -> None:
        hand = FakeHand()

        async def scenario():
            controller = WujiController(hand, transport="sdk", max_tracking_error=0.2)
            await controller.start()
            hand.follow = False
            hand.q = [0.1] * 3 + [0.6] + [0.1] * 16
            with self.assertRaisesRegex(RuntimeError, "tracking error 0.500"):
                await controller.wait(1.0)
            await controller.stop()

        run(scenario())
        self.assertIn("disable", hand.calls)

    def test_tracking_guard_is_off_by_default(self) -> None:
        hand = FakeHand()

        async def scenario():
            controller = WujiController(hand, transport="sdk")
            await controller.start()
            hand.follow = False
            hand.q = [0.1] * 3 + [1.0] + [0.1] * 16  # pushed 0.9 rad away
            await controller.wait(0.1)
            self.assertIsNone(controller.error)
            self.assertAlmostEqual(controller.stats["max_tracking_error"], 0.9)
            hand.streaming = False  # other guards still apply
            with self.assertRaisesRegex(RuntimeError, "no joint_states"):
                await controller.wait(1.0)
            await controller.stop()

        run(scenario())
        self.assertNotIn("emergency_stop", hand.calls)

    def test_stale_joint_states_disable(self) -> None:
        hand = FakeHand()

        async def scenario():
            controller = WujiController(hand, transport="sdk")
            await controller.start()
            hand.streaming = False
            with self.assertRaisesRegex(RuntimeError, "no joint_states"):
                await controller.wait(1.0)
            await controller.stop()

        run(scenario())
        self.assertIn("disable", hand.calls)

    def test_warnings_are_collected_and_faults_disable(self) -> None:
        hand = FakeHand()

        async def scenario():
            controller = WujiController(hand, transport="sdk")
            await controller.start()
            hand.error_code = 0x0007  # Stall: a firmware Warning
            await controller.wait(0.05)
            self.assertIn("Stall", controller.warnings)
            hand.error_code = 0x2102  # Overcurrent: ImmediateStop
            with self.assertRaisesRegex(RuntimeError, "Overcurrent"):
                await controller.wait(1.0)
            await controller.stop()

        run(scenario())

    def test_refuses_to_start_unless_every_joint_is_ready(self) -> None:
        hand = FakeHand()
        hand.ext_state = 2

        async def scenario():
            controller = WujiController(hand, transport="sdk")
            with self.assertRaisesRegex(RuntimeError, "not Ready"):
                await controller.start()

        run(scenario())
        self.assertEqual(hand.calls, [])
        self.assertTrue(all(s.closed for s in hand.subscriptions))

    def test_feed_forward_is_set_before_gains_and_enable(self) -> None:
        for on, expected in ((True, [("set_constant", 3), ("set_position_table", True)]),
                             (False, [("set_constant", 0), ("set_position_table", False)])):
            hand = FakeHand()

            async def scenario():
                controller = WujiController(hand, transport="sdk", feed_forward=on)
                await controller.start()
                await controller.stop()

            run(scenario())
            self.assertEqual(hand.calls[:3], expected + [("effort_limit", 0.5)])
            self.assertIn("enable", hand.calls)

        hand = FakeHand()  # the default leaves the firmware setting alone

        async def untouched():
            controller = WujiController(hand, transport="sdk")
            await controller.start()
            await controller.stop()

        run(untouched())
        self.assertFalse([c for c in hand.calls if isinstance(c, tuple) and c[0] in ("set_constant", "set_position_table")])

    def test_dry_run_changes_nothing_on_the_hand(self) -> None:
        hand = FakeHand()

        async def scenario():
            controller = WujiController(hand, transport="sdk", dry_run=True)
            await controller.start()
            target = controller.initial_qpos.copy()
            target[19] -= 0.05
            await controller.move(target)
            await controller.stop()
            return controller

        controller = run(scenario())
        self.assertEqual((hand.calls, hand.sent), ([], []))
        self.assertGreater(controller.stats["ticks"], 10)

    def test_target_validation(self) -> None:
        with self.assertRaises(ValueError):
            WujiController(effort_limit=2.0)
        with self.assertRaises(ValueError):
            WujiController(kp=-1.0)
        hand = FakeHand()

        async def scenario():
            controller = WujiController(hand, transport="sdk", dry_run=True)
            with self.assertRaisesRegex(RuntimeError, "start"):
                controller.set_now("q_desired", np.zeros(20))
            await controller.start()
            with self.assertRaisesRegex(ValueError, "outside"):
                controller.set_now("q_desired", np.full(20, 2.5))
            with self.assertRaisesRegex(ValueError, "effort_limit"):
                controller.set_now("torque", np.full(20, 0.6))
            with self.assertRaisesRegex(ValueError, "shape"):
                controller.set_now("q_desired", np.zeros(19))
            await controller.stop()

        run(scenario())


if __name__ == "__main__":
    unittest.main()
