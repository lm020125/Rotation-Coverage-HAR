from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def create_synthetic_dataset(
    output_dir: str | Path,
    participants: int = 4,
    windows_per_participant: int = 24,
    length: int = 100,
    channels: int = 6,
    classes: int = 4,
    seed: int = 7,
) -> Path:
    """Create tiny role-structured data for smoke tests; never use it as research evidence."""
    output_dir = Path(output_dir)
    window_dir = output_dir / "windows"
    window_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    time = np.linspace(0, 2 * np.pi, length, dtype=np.float32)
    for p_idx in range(participants):
        participant = f"P{p_idx + 1:02d}"
        left_dominant = p_idx % 2 == 1
        for w_idx in range(windows_per_participant):
            label = w_idx % classes
            frequency = label + 1
            actor = np.stack(
                [np.sin(frequency * time + 0.15 * c) for c in range(channels)], axis=-1
            ).astype(np.float32)
            support = 0.25 * np.stack(
                [np.cos(frequency * time + 0.1 * c) for c in range(channels)], axis=-1
            ).astype(np.float32)
            actor += rng.normal(0, 0.08, actor.shape).astype(np.float32)
            support += rng.normal(0, 0.08, support.shape).astype(np.float32)
            use_left = left_dominant ^ (w_idx % 5 == 0)
            left, right = (actor, support) if use_left else (support, actor)
            path = window_dir / f"{participant}_{w_idx:04d}.npz"
            np.savez_compressed(path, left=left, right=right)
            rows.append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "participant": participant,
                    "label": label,
                    "label_name": f"class_{label}",
                }
            )
    manifest = output_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False, encoding="utf-8")
    return manifest


def create_four_limb_synthetic_dataset(
    output_dir: str | Path,
    participants: int = 6,
    windows_per_participant: int = 12,
    length: int = 128,
    classes: int = 4,
    seed: int = 11,
) -> Path:
    """Create four-limb NPZ windows for upgraded-pipeline smoke tests."""
    output_dir = Path(output_dir)
    window_dir = output_dir / "windows"
    window_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    time = np.linspace(0, 2 * np.pi, length, dtype=np.float32)
    for p_idx in range(participants):
        participant = f"P{p_idx + 1:02d}"
        for w_idx in range(windows_per_participant):
            label = w_idx % classes
            base = label + 1
            arm_actor = np.stack([np.sin(base * time + c * 0.2) for c in range(3)], -1)
            arm_support = 0.3 * np.stack([np.cos(base * time + c * 0.1) for c in range(3)], -1)
            leg_actor = np.stack([np.sin((base + 0.5) * time + c * 0.15) for c in range(3)], -1)
            leg_support = 0.45 * np.stack(
                [np.cos((base + 0.5) * time + c * 0.1) for c in range(3)], -1
            )
            signals = [arm_actor, arm_support, leg_actor, leg_support]
            signals = [
                (signal + rng.normal(0, 0.06, signal.shape)).astype(np.float32)
                for signal in signals
            ]
            if (p_idx + w_idx) % 2:
                signals[0], signals[1] = signals[1], signals[0]
            if (p_idx + label) % 2:
                signals[2], signals[3] = signals[3], signals[2]
            path = window_dir / f"{participant}_{w_idx:04d}.npz"
            np.savez_compressed(
                path,
                left_wrist=signals[0],
                right_wrist=signals[1],
                left_ankle=signals[2],
                right_ankle=signals[3],
            )
            rows.append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "participant": participant,
                    "session": "S01",
                    "center": w_idx,
                    "label": label,
                    "label_name": f"class_{label}",
                }
            )
    manifest = output_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False, encoding="utf-8")
    return manifest
