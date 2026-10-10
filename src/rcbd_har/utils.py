from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def save_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def participant_sort_key(value: str) -> tuple[object, ...]:
    """Sort participant identifiers naturally (0, 1, 2, ..., 10)."""
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("-inf")
        self.bad_epochs = 0

    def update(self, score: float) -> bool:
        if score > self.best:
            self.best = score
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience
