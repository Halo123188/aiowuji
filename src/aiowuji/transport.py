"""Joint telemetry and command streams for the control loop.

``SdkTransport`` uses only the public ``wuji_sdk`` API, so it runs on the
official Linux SDK and on the macOS port alike. ``NativeTransport`` is a faster
path available with the macOS port (wuji-macos): it drains the native receivers
without building Python value objects, decodes ``joint_states`` with ``struct``
(about 1 us instead of about 70 us per frame), and hands commands to the native
publisher as one float32[60].
"""
from __future__ import annotations

import ctypes
from collections import namedtuple
import struct

from .hand import DOF, dense_index

JointState = namedtuple("JointState", "seq timestamp_us qpos qvel effort")
JointStatus = namedtuple("JointStatus", "nid ext_state error_code")


def _dense(entries):
    """(nid, position, velocity, effort) rows -> dense lists; absent joints are None."""
    q, qd, effort = [None] * DOF, [None] * DOF, [None] * DOF
    for nid, position, velocity, current in entries:
        j = dense_index(nid)
        q[j], qd[j], effort[j] = position, velocity, current
    return q, qd, effort


class SdkTransport:
    name = "sdk"
    # Every diagnostics frame is decoded on receipt, so poll each tick rather
    # than decoding a burst of queued frames at once.
    status_every = 1

    def __init__(self, sdk, states, diagnostics, publisher=None):
        self._command = sdk.JointCommand
        self.states, self.diagnostics, self.publisher = states, diagnostics, publisher
        self._lagged = 0

    @property
    def lagged_frames(self):
        # The macOS port's recv() skips a backlog and counts it on the subscription.
        counted = 0
        for subscription in (self.states, self.diagnostics):
            try:
                counted += int(getattr(subscription, "lagged_frames", 0))
            except Exception:
                pass
        return self._lagged + counted

    def _drain(self, subscription):
        latest = None
        while True:
            try:
                frame = subscription.recv()
            except Exception as error:
                # A lagged receiver has skipped ahead; anything else is fatal.
                if "lag" not in str(error).lower():
                    raise
                self._lagged += getattr(error, "lagged_frames", 1)
                continue
            if frame is None:
                return latest
            latest = frame

    def latest_state(self):
        frame = self._drain(self.states)
        if frame is None:
            return None
        rows = ((j.nid, j.position, j.velocity, j.effort) for j in frame.joints)
        return JointState(frame.header.seq, frame.header.timestamp_us, *_dense(rows))

    def latest_status(self):
        frame = self._drain(self.diagnostics)
        if frame is None:
            return None
        return [JointStatus(j.nid, j.status_word.ext_state, j.error_code_current) for j in frame.joints]

    def send(self, qpos, qvel, effort):
        self.publisher.send([self._command(p, v, e) for p, v, e in zip(qpos, qvel, effort)])

    def close(self):
        errors = []
        for resource in (self.publisher, self.states, self.diagnostics):
            try:
                if resource is not None:
                    resource.close()
            except Exception as error:
                errors.append(str(error))
        return errors


_ENTRY = struct.Struct("<Bfff")


def _varint(buf, i):
    value = shift = 0
    while True:
        byte = buf[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, i
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def decode_joint_states(payload: bytes) -> JointState:
    """Decode a Postcard JointStateFrame (schema 330).

    Wire order: header {seq varint, timestamp_us varint, frame_id string},
    num_joints u8, then a Vec of {nid u8, position f32, velocity f32, effort f32}.
    """
    try:
        seq, i = _varint(payload, 0)
        stamp, i = _varint(payload, i)
        length, i = _varint(payload, i)
        count, i = _varint(payload, i + length + 1)  # skip frame_id and num_joints
    except IndexError:
        raise ValueError("truncated joint_states payload") from None
    if i + _ENTRY.size * count != len(payload):
        raise ValueError("malformed joint_states payload")
    return JointState(seq, stamp, *_dense(_ENTRY.iter_unpack(payload[i:])))


class _RawReceiver:
    def __init__(self, subscription, schema_id):
        backend = subscription._backend
        descriptor = backend._subscriptions[subscription._handle]
        if (descriptor["schema_id"], descriptor["serde_format"]) != (schema_id, 0):
            raise RuntimeError(f"expected Postcard schema {schema_id}, got {descriptor}")
        self._handle = subscription._handle
        self._recv = backend.device_lib.wuji_subscription_try_recv
        self._free = backend.device_lib.wuji_device_buffer_free
        self._out, self._length, self._lag = ctypes.c_void_p(), ctypes.c_size_t(), ctypes.c_uint64()
        self._error = ctypes.create_string_buffer(256)
        self.lagged_frames = 0

    def drain(self):
        latest = None
        while True:
            code = self._recv(self._handle, ctypes.byref(self._out), ctypes.byref(self._length),
                              ctypes.byref(self._lag), self._error, len(self._error))
            if code == 0:
                try:
                    latest = ctypes.string_at(self._out, self._length.value) if self._out else b""
                finally:
                    self._free(self._out)
            elif code == 1:
                return latest
            elif code == 4:
                self.lagged_frames += self._lag.value
            else:
                raise RuntimeError(f"native receive failed ({code}): {self._error.value.decode()}")


class NativeTransport(SdkTransport):
    name = "native"
    # Draining raw frames is cheap and only the newest is decoded.
    status_every = 10

    def __init__(self, sdk, states, diagnostics, publisher=None):
        super().__init__(sdk, states, diagnostics, publisher)
        from wuji_sdk import _codec  # present only in wuji-macos

        self._decode = _codec.decode
        self._states = _RawReceiver(states, 330)
        self._diagnostics = _RawReceiver(diagnostics, 332)
        if publisher is not None:
            backend = publisher._backend
            entry = backend._publishers[publisher._handle]
            if not entry["typed"] or entry["descriptor"]["schema_id"] != 315:
                raise RuntimeError("expected the typed joint_command publisher (schema 315)")
            self._send = backend.device_lib.wuji_publisher_send_joint_commands
            self._buffer = (ctypes.c_float * (3 * DOF))()
            self._send_error = ctypes.create_string_buffer(256)

    @property
    def lagged_frames(self):
        return self._states.lagged_frames + self._diagnostics.lagged_frames

    def latest_state(self):
        payload = self._states.drain()
        return None if payload is None else decode_joint_states(payload)

    def latest_status(self):
        payload = self._diagnostics.drain()
        if payload is None:
            return None
        frame = self._decode(332, payload)
        return [JointStatus(j["nid"], j["status_word"] & 3, j["error_code_current"]) for j in frame["joints"]]

    def send(self, qpos, qvel, effort):
        buffer = self._buffer
        buffer[0::3], buffer[1::3], buffer[2::3] = qpos, qvel, effort
        code = self._send(self.publisher._handle, buffer, len(buffer), self._send_error, len(self._send_error))
        if code:
            raise RuntimeError(f"joint command send failed ({code}): {self._send_error.value.decode()}")


def _native_capable(subscription) -> bool:
    backend = getattr(subscription, "_backend", None)
    return hasattr(backend, "device_lib") and hasattr(backend, "_subscriptions")


def open_transport(sdk, hand, *, publish: bool, kind: str = "auto"):
    """Subscribe to joint_states/joint_diagnostics and optionally open joint_command."""
    if kind not in ("auto", "native", "sdk"):
        raise ValueError("transport must be 'auto', 'native', or 'sdk'")
    states = hand.joint_states().subscribe()
    diagnostics = publisher = None
    try:
        diagnostics = hand.joint_diagnostics().subscribe()
        publisher = hand.joint_command().publish() if publish else None
        native = _native_capable(states)
        if kind == "native" and not native:
            raise RuntimeError("the native transport requires the wuji-macos SDK")
        cls = NativeTransport if native and kind != "sdk" else SdkTransport
        return cls(sdk, states, diagnostics, publisher)
    except BaseException:
        for resource in (publisher, diagnostics, states):
            if resource is not None:
                resource.close()
        raise
