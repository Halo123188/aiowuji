"""aiowuji command line: read-only discovery and status."""
from __future__ import annotations

import argparse
import time

import numpy as np

from .hand import JOINT_NAMES, JOINT_NIDS, connect, import_sdk
from .transport import open_transport

_EXT_STATES = {0: "Init", 1: "Ready", 2: "Enabled", 3: "Stopped"}


def scan() -> int:
    sdk = import_sdk()
    devices = sdk.SdkManager.instance().scan()
    if not devices:
        print("No Wuji devices found")
        return 1
    for device in devices:
        print(f"{device.sn}  {device.device_type}  {device.address}")
    return 0


def status(hand_sn: str | None) -> int:
    sdk = import_sdk()
    manager, hand = connect(hand_sn, device_name="aiowuji_status")
    try:
        info = hand.info
        print(f"Serial:    {hand.serial_number}")
        print(f"Firmware:  {info.firmware_version if info is not None else 'unknown'}")
        print(f"Side:      {hand.handedness().get()}")
        print(f"Online:    {hand.online_joints_count().get()} joints")
        gains, limits = hand.mit_params().get(), hand.effort_limit().get()
        transport = open_transport(sdk, hand, publish=False)
        try:
            deadline, state, joints = time.monotonic() + 1.0, None, None
            while (state is None or joints is None) and time.monotonic() < deadline:
                state = transport.latest_state() or state
                joints = transport.latest_status() or joints
                time.sleep(0.005)
        finally:
            transport.close()
        by_nid = {j.nid: j for j in joints or []}
        print(f"Transport: {transport.name}\n")
        print(f"{'joint':<10}{'pos(deg)':>9}  {'state':<8}{'error':<7}{'kp':>6}{'kd':>7}{'limit(A)':>9}")
        for index, (name, nid) in enumerate(zip(JOINT_NAMES, JOINT_NIDS)):
            joint = by_nid.get(nid)
            position = state.qpos[index] if state is not None else None
            gain = gains[index]
            print(f"{name:<10}"
                  f"{'-' if position is None else f'{np.rad2deg(position):9.2f}':>9}  "
                  f"{_EXT_STATES.get(joint.ext_state, '?') if joint else 'offline':<8}"
                  f"{f'0x{joint.error_code:04x}' if joint and joint.error_code else '-':<7}"
                  f"{'-' if gain is None else f'{gain.kp:6.2f}':>6}"
                  f"{'-' if gain is None else f'{gain.kd:7.3f}':>7}"
                  f"{'-' if limits[index] is None else f'{limits[index]:9.2f}':>9}")
    finally:
        manager.disconnect("aiowuji_status")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="aiowuji", description="Wuji Hand 2 tools (read-only)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scan", help="list discovered Wuji devices")
    status_parser = commands.add_parser("status", help="show hand, joint, and gain status")
    status_parser.add_argument("hand_sn", nargs="?", help="serial (default: last used or only hand)")
    args = parser.parse_args(argv)
    return scan() if args.command == "scan" else status(args.hand_sn)


if __name__ == "__main__":
    raise SystemExit(main())
