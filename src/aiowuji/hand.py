"""Wuji Hand 2 constants and SDK connection helpers."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import sys

import numpy as np

from .config import get_last_hand_sn, save_last_hand_sn

DOF = 20
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
JOINT_NAMES = tuple(f"{finger}_S{slot}" for finger in FINGERS for slot in range(1, 5))
# Telemetry NID of each dense joint index: five slots per finger bus, joints in 1-4.
JOINT_NIDS = tuple(bus * 5 + slot for bus in range(5) for slot in range(1, 5))

# Documented joint ranges in degrees, thumb then four fingers:
# https://docs.wuji.tech/docs/en/wuji-hand/latest/control-guide/
_THUMB_LIMITS_DEG = ((-68, 74), (-85, 40), (-60, 90), (-60, 90))
_FINGER_LIMITS_DEG = ((-60, 90), (-40, 40), (-60, 120), (-60, 90))
JOINT_LIMITS = np.deg2rad(np.array(_THUMB_LIMITS_DEG + _FINGER_LIMITS_DEG * 4, dtype=float))


def dense_index(nid: int) -> int:
    """Map a sparse telemetry NID (five slots per finger bus, joints in slots 1-4)
    to the dense joint index 0..19 used by commands and gains."""
    bus, slot = divmod(int(nid), 5)
    if bus >= 5 or not 1 <= slot <= 4:
        raise ValueError(f"unexpected joint NID {nid}")
    return bus * 4 + slot - 1


def outside_limits(qpos) -> list[int]:
    q = np.asarray(qpos, dtype=float)
    return [int(j) for j in np.flatnonzero((q < JOINT_LIMITS[:, 0]) | (q > JOINT_LIMITS[:, 1]))]


def import_sdk():
    """Import ``wuji_sdk`` from the official SDK (Linux) or wuji-macos (Apple Silicon).

    On macOS the port keeps SDK state under ``$WUJI_MACOS_STATE_DIR/.wuji``;
    default it to the home directory so users and calibrations live in
    ``~/.wuji`` exactly as with the official SDK on Linux.
    """
    if sys.platform == "darwin":
        os.environ.setdefault("WUJI_MACOS_STATE_DIR", str(Path.home()))
    try:
        import wuji_sdk
    except ModuleNotFoundError as error:
        if sys.platform == "darwin" and platform.machine() != "arm64":
            raise ImportError("aiowuji on macOS requires Apple Silicon (wuji-macos)") from error
        raise ImportError(
            "wuji_sdk is not installed: `pip install aiowuji` installs wuji-sdk on Linux "
            "and wuji-macos on Apple Silicon"
        ) from error
    return wuji_sdk


def connect(hand_sn: str | None = None, *, device_name: str = "aiowuji"):
    """Connect a Wuji Hand 2 and return ``(manager, hand)``.

    Without a serial, reuse the last connected serial; otherwise require exactly
    one discovered Hand 2. A successful connection is saved as the last serial.
    """
    sdk = import_sdk()
    manager = sdk.SdkManager.instance()
    serial = hand_sn or get_last_hand_sn()
    if serial is None:
        hands = [d for d in manager.scan() if d.device_type == sdk.DeviceType.WujiHand2]
        if len(hands) != 1:
            raise RuntimeError(f"Found {len(hands)} Wuji Hand 2 devices; pass a serial number")
        serial = hands[0].sn
    hand = manager.connect(
        sn=serial,
        device_name=device_name,
        options=sdk.ConnectOptions(enable_bridge=False),
    )
    save_last_hand_sn(hand.serial_number)
    return manager, hand
