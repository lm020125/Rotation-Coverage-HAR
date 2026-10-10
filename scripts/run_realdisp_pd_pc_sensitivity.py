from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.config import load_config  # noqa: E402


SEEDS = (256, 3407, 114514)
TRAIN_PARTICIPANTS = tuple(f"subject{value}" for value in range(1, 11))
VALIDATION_PARTICIPANTS = ("subject11", "subject12")
FROZEN_PROFILE_SHA256 = (
    "b05eedb46aa01cff0c2331b09bc3ca4911821a0ef7c8291c0807dd19c555b2b7"
)


@dataclass(frozen=True)
class SensitivitySetting:
    name: str
    augmentation_probability: float
    consistency_weight: float
    reuse_frozen_center: bool = False


SETTINGS = (
    SensitivitySetting("p025_l010", 0.25, 0.10),
    SensitivitySetting("p050_l005", 0.50, 0.05),
    SensitivitySetting("p050_l010", 0.50, 0.10, reuse_frozen_center=True),
    SensitivitySetting("p050_l020", 0.50, 0.20),
    SensitivitySetting("p075_l010", 0.75, 0.10),
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_setting_config(
    base_config: dict[str, Any], setting: SensitivitySetting
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config["training"]["quality_augmentation_probability"] = float(
        setting.augmentation_probability
    )
    config["loss"]["perturbation_prediction_consistency"] = float(
        setting.consistency_weight
    )
    return config


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def metric_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
    }


def source_root(
    output_root: Path,
    frozen_center_root: Path,
    setting: SensitivitySetting,
    seed: int,
) -> Path:
    if setting.reuse_frozen_center:
        return frozen_center_root / "pd_pc" / f"seed{seed}"
    return output_root / setting.name / f"seed{seed}"


def aggregate(output_root: Path, frozen_center_root: Path) -> dict[str, Any]:
    settings_payload: dict[str, Any] = {}
    for setting in SETTINGS:
        rows = []
        for seed in SEEDS:
            source = source_root(output_root, frozen_center_root, setting, seed)
            self_path = (
                output_root
                / setting.name
                / f"seed{seed}"
                / "validation_self_subjects11_12"
                / "metrics.json"
            )
            validation_path = source / "validation_metrics.json"
            if not validation_path.exists() or not self_path.exists():
                continue
            validation = read_json(validation_path)
            self_metrics = read_json(self_path)
            rows.append(
                {
                    "seed": seed,
                    "validation_ideal_macro_f1": float(validation["clean"]["macro_f1"]),
                    "validation_rotation_stress_macro_f1": float(
                        validation["rotation_stress"]["macro_f1_mean"]
                    ),
                    "validation_self_macro_f1": float(self_metrics["macro_f1"]),
                    "validation_ideal_accuracy": float(validation["clean"]["accuracy"]),
                    "validation_self_accuracy": float(self_metrics["accuracy"]),
                    "source_checkpoint": str(source / "best.pt"),
                }
            )
        payload: dict[str, Any] = {"setting": asdict(setting), "runs": rows}
        if rows:
            for key in (
                "validation_ideal_macro_f1",
                "validation_rotation_stress_macro_f1",
                "validation_self_macro_f1",
                "validation_ideal_accuracy",
                "validation_self_accuracy",
            ):
                payload[key] = metric_summary([float(row[key]) for row in rows])
        settings_payload[setting.name] = payload
    return {
        "method": "PD-PC frozen post-hoc validation-only sensitivity analysis",
        "protocol": {
            "purpose": "descriptive robustness; not hyperparameter reselection",
            "supervised_train": list(TRAIN_PARTICIPANTS),
            "validation_ideal_and_self": list(VALIDATION_PARTICIPANTS),
            "final_test_accessed": False,
            "seeds": list(SEEDS),
            "frozen_center": {"probability": 0.50, "consistency_weight": 0.10},
            "execution": "strictly sequential and resumable",
        },
        "settings": settings_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen validation-only PD-PC hyperparameter sensitivity."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stress-repeats", type=int, default=3)
    parser.add_argument(
        "--base-config",
        default="configs/realdisp_rcbd_rotation_pred_consistency.yaml",
    )
    parser.add_argument(
        "--placement-profile",
        default=(
            "/root/autodl-tmp/experiments/"
            "realdisp_pd_pc_component_ablation_20260718/"
            "placement_profiles_subjects1_10.json"
        ),
    )
    parser.add_argument(
        "--frozen-center-root",
        default=(
            "/root/autodl-tmp/experiments/"
            "realdisp_pd_pc_component_ablation_20260718"
        ),
    )
    parser.add_argument(
        "--self-manifest",
        default=(
            "/root/autodl-tmp/realdisp_project/data/realdisp_distal_33/"
            "manifest_self.csv"
        ),
    )
    args = parser.parse_args()

    if args.stress_repeats < 1:
        raise ValueError("stress-repeats must be positive")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    profile = Path(args.placement_profile).resolve()
    profile_sha = file_sha256(profile)
    if profile_sha != FROZEN_PROFILE_SHA256:
        raise ValueError(
            f"Frozen placement profile SHA mismatch: {profile_sha}"
        )
    frozen_center_root = Path(args.frozen_center_root).resolve()
    base_config = load_config(args.base_config)
    generated_configs = output_root / "generated_configs"
    for setting in SETTINGS:
        write_json(
            build_setting_config(base_config, setting),
            generated_configs / f"{setting.name}.json",
        )

    total = len(SETTINGS) * len(SEEDS)
    position = 0
    for setting in SETTINGS:
        setting_config = generated_configs / f"{setting.name}.json"
        for seed in SEEDS:
            position += 1
            run_root = output_root / setting.name / f"seed{seed}"
            source = source_root(output_root, frozen_center_root, setting, seed)
            write_json(
                {
                    "position": position,
                    "total": total,
                    "setting": setting.name,
                    "seed": seed,
                    "state": "running",
                    "center_checkpoint_reused": setting.reuse_frozen_center,
                },
                output_root / "queue_status.json",
            )
            print(
                f"[{position}/{total}] setting={setting.name} seed={seed} "
                f"reuse_center={setting.reuse_frozen_center}",
                flush=True,
            )
            if not setting.reuse_frozen_center and not (
                source / "validation_metrics.json"
            ).exists():
                run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "run_realdisp_validation_diagnostic.py"),
                        "--config",
                        str(setting_config),
                        "--output",
                        str(source),
                        "--device",
                        args.device,
                        "--seed",
                        str(seed),
                        "--stress-repeats",
                        str(args.stress_repeats),
                        "--placement-profile",
                        str(profile),
                    ]
                )
            checkpoint = source / "best.pt"
            validation_metrics = source / "validation_metrics.json"
            if not checkpoint.exists() or not validation_metrics.exists():
                raise FileNotFoundError(
                    f"Missing frozen source artifacts for {setting.name} seed{seed}"
                )
            self_output = run_root / "validation_self_subjects11_12"
            if not (self_output / "metrics.json").exists():
                run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "evaluate_realdisp_checkpoint.py"),
                        "--checkpoint",
                        str(checkpoint),
                        "--manifest",
                        args.self_manifest,
                        "--output",
                        str(self_output),
                        "--device",
                        args.device,
                        "--participants",
                        *VALIDATION_PARTICIPANTS,
                        "--num-workers",
                        "4",
                    ]
                )
            write_json(
                {
                    "setting": asdict(setting),
                    "seed": seed,
                    "source_root": str(source),
                    "profile_sha256": profile_sha,
                    "final_test_accessed": False,
                },
                run_root / "sensitivity_protocol.json",
            )
            write_json(
                aggregate(output_root, frozen_center_root),
                output_root / "summary.partial.json",
            )

    summary = aggregate(output_root, frozen_center_root)
    write_json(summary, output_root / "summary.json")
    write_json(
        {
            "position": total,
            "total": total,
            "state": "completed",
            "final_test_accessed": False,
        },
        output_root / "queue_status.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
