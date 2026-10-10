from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and validate the required top-level fields."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    for key in ("data", "model", "training", "loss"):
        if key not in config:
            raise ValueError(f"Missing configuration section: {key}")
    return config


def with_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Return a copied config with dotted-key overrides."""
    updated = deepcopy(config)
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        cursor = updated
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return updated
