from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def config_path() -> Path:
    override = os.environ.get("AIOWUJI_CONFIG")
    if override:
        return Path(override).expanduser()

    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return base / "aiowuji" / "config.json"


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid aiowuji config at {path}: expected a JSON object")
    return data


def get_last_hand_sn() -> str | None:
    value = load_config().get("last_hand_sn")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def save_last_hand_sn(hand_sn: str) -> None:
    serial = hand_sn.strip()
    if not serial:
        raise ValueError("hand_sn must not be empty")

    path = config_path()
    config = load_config()
    config["last_hand_sn"] = serial

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)
