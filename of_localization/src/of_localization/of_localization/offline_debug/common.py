from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def package_root() -> Path:
    """Return the package root directory."""
    return Path(__file__).resolve().parents[2]


def default_params_path() -> Path:
    """Return the default OF params file."""
    return package_root() / "config" / "params.yaml"


def load_ros_params(params_file: str | Path | None = None) -> dict[str, Any]:
    """Load the package ROS parameter dictionary from YAML."""
    params_path = Path(params_file) if params_file is not None else default_params_path()
    with params_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not data:
        return {}
    if "/**" in data and "ros__parameters" in data["/**"]:
        return dict(data["/**"]["ros__parameters"])
    if "ros_parameters" in data:
        return dict(data["ros_parameters"])
    return dict(data)
