"""YAML config loading."""
from __future__ import annotations

from pathlib import Path

import yaml

from .env import EnvConfig


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["env"] = EnvConfig(**cfg.get("env", {}))
    return cfg
