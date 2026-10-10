from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_paired_participants import (  # noqa: E402
    exact_sign_flip_pvalue,
    paired_bootstrap_interval,
)


def test_paired_bootstrap_is_reproducible() -> None:
    values = np.array([0.01, 0.02, -0.01, 0.03])
    first = paired_bootstrap_interval(values, samples=1000, seed=7)
    second = paired_bootstrap_interval(values, samples=1000, seed=7)
    assert first == second
    assert first[0] <= values.mean() <= first[1]


def test_exact_sign_flip_detects_uniform_positive_shift() -> None:
    values = np.ones(10)
    assert exact_sign_flip_pvalue(values) == 2 / (2**10)
