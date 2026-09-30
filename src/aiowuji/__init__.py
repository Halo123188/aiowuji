from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .controller import MAX_EFFORT_LIMIT, WujiController
from .hand import DOF, FINGERS, JOINT_LIMITS, JOINT_NAMES, JOINT_NIDS, connect, dense_index, import_sdk
from .transport import JointState, JointStatus, NativeTransport, SdkTransport, open_transport

try:
    __version__ = version("aiowuji")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "WujiController",
    "MAX_EFFORT_LIMIT",
    "DOF",
    "FINGERS",
    "JOINT_LIMITS",
    "JOINT_NAMES",
    "JOINT_NIDS",
    "JointState",
    "JointStatus",
    "NativeTransport",
    "SdkTransport",
    "connect",
    "dense_index",
    "import_sdk",
    "open_transport",
]
