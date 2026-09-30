from __future__ import annotations

import struct
from types import SimpleNamespace
import unittest

from aiowuji.hand import JOINT_NIDS, dense_index
from aiowuji.transport import SdkTransport, decode_joint_states


def _varint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def joint_states_payload(q, *, seq=300, stamp=1789936356771617, frame_id=b"r_wrist", order=range(20)):
    order = list(order)
    out = _varint(seq) + _varint(stamp) + _varint(len(frame_id)) + frame_id
    out += bytes([len(order)]) + _varint(len(order))
    for j in order:
        out += struct.pack("<Bfff", JOINT_NIDS[j], q[j], 0.25 * j, -0.5 * j)
    return out


class HandTests(unittest.TestCase):
    def test_nids_map_to_dense_command_order(self) -> None:
        self.assertEqual([dense_index(nid) for nid in JOINT_NIDS], list(range(20)))
        for nid in (0, 5, 10, 15, 20, 25):
            with self.assertRaises(ValueError):
                dense_index(nid)


class DecodeTests(unittest.TestCase):
    def test_decodes_full_frame_in_any_order(self) -> None:
        q = [0.125 * j - 1.0 for j in range(20)]
        order = [7, 19, 0] + [j for j in range(20) if j not in (7, 19, 0)]
        state = decode_joint_states(joint_states_payload(q, seq=2**31, frame_id=b"x" * 200, order=order))
        self.assertEqual((state.seq, state.timestamp_us), (2**31, 1789936356771617))
        self.assertEqual(state.qpos, q)
        self.assertEqual(state.qvel[4], 1.0)
        self.assertEqual(state.effort[4], -2.0)

    def test_absent_joints_are_none(self) -> None:
        state = decode_joint_states(joint_states_payload([0.5] * 20, order=[0, 19]))
        self.assertEqual((state.qpos[0], state.qpos[19], state.qpos.count(None)), (0.5, 0.5, 18))

    def test_malformed_payloads_raise(self) -> None:
        payload = joint_states_payload([0.0] * 20)
        for bad in (payload[:-1], payload + b"\0", payload[:3]):
            with self.assertRaises(ValueError):
                decode_joint_states(bad)

    def test_matches_macos_port_codec_when_available(self) -> None:
        try:
            from wuji_sdk import _codec
        except ImportError:
            self.skipTest("wuji_sdk._codec is specific to the macOS port")
        payload = joint_states_payload([0.1 * j for j in range(20)], order=range(19, -1, -1))
        reference = _codec.decode(330, payload)
        state = decode_joint_states(payload)
        for joint in reference["joints"]:
            j = dense_index(joint["nid"])
            self.assertEqual((state.qpos[j], state.qvel[j], state.effort[j]),
                             (joint["position"], joint["velocity"], joint["effort"]))


class Lagged(Exception):
    def __init__(self, frames: int) -> None:
        super().__init__(f"Stream lagged by {frames} frame(s)")
        self.lagged_frames = frames


class ScriptedSubscription:
    def __init__(self, items) -> None:
        self.items = list(items)
        self.closed = False

    def recv(self):
        item = self.items.pop(0) if self.items else None
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


def state_frame(seq: int, value: float):
    joints = [SimpleNamespace(nid=nid, position=value, velocity=0.0, effort=0.0) for nid in JOINT_NIDS]
    return SimpleNamespace(header=SimpleNamespace(seq=seq, timestamp_us=seq), joints=joints)


class SdkTransportTests(unittest.TestCase):
    def test_drains_to_newest_frame_and_survives_lag(self) -> None:
        states = ScriptedSubscription([state_frame(1, 0.1), Lagged(4), state_frame(2, 0.2)])
        diagnostics = ScriptedSubscription([SimpleNamespace(joints=[
            SimpleNamespace(nid=1, status_word=SimpleNamespace(ext_state=2), error_code_current=7)])])
        transport = SdkTransport(SimpleNamespace(JointCommand=None), states, diagnostics)
        state = transport.latest_state()
        self.assertEqual((state.seq, state.qpos[0], transport.lagged_frames), (2, 0.2, 4))
        self.assertIsNone(transport.latest_state())
        self.assertEqual(transport.latest_status()[0], (1, 2, 7))

    def test_other_stream_errors_propagate(self) -> None:
        transport = SdkTransport(SimpleNamespace(JointCommand=None),
                                 ScriptedSubscription([RuntimeError("StreamClosed")]), None)
        with self.assertRaisesRegex(RuntimeError, "StreamClosed"):
            transport.latest_state()

    def test_send_builds_twenty_joint_commands(self) -> None:
        sent = []
        publisher = SimpleNamespace(send=sent.append, close=lambda: None)
        transport = SdkTransport(SimpleNamespace(JointCommand=lambda p, v, e: (p, v, e)),
                                 ScriptedSubscription([]), ScriptedSubscription([]), publisher)
        transport.send([1.0] * 20, [2.0] * 20, [3.0] * 20)
        self.assertEqual(sent, [[(1.0, 2.0, 3.0)] * 20])
        self.assertEqual(transport.close(), [])


if __name__ == "__main__":
    unittest.main()
