from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.config import load_config  # noqa: E402


SEEDS = (256, 3407, 114514)
TRAIN_PARTICIPANTS = tuple(f"subject{value}" for value in range(1, 11))
VALIDATION_PARTICIPANTS = ("subject11", "subject12")
AS_RELEASED_PARTICIPANTS = tuple(f"subject{value}" for value in range(13, 18))
VALID_SIGNAL_PARTICIPANTS = tuple(f"subject{value}" for value in range(14, 18))
MUTUAL_PARTICIPANT = ("subject15",)
FROZEN_PROFILE_SHA256 = (
    "b05eedb46aa01cff0c2331b09bc3ca4911821a0ef7c8291c0807dd19c555b2b7"
)

RCBD_ABLATIONS = {
    "fmc_angle_pc": "configs/realdisp_fmc_angle_pc.yaml",
    "fmc_angle_gain_pc": "configs/realdisp_fmc_angle_gain_pc.yaml",
    "fmc_full_no_pc": "configs/realdisp_fmc_full_no_pc.yaml",
    "fmc_global_pc": "configs/realdisp_fmc_global_pc.yaml",
}
CROSS_BACKBONE = {
    "deepconvlstm_fmc_pc": "configs/realdisp_deepconvlstm_fmc_pc.yaml",
}
EXISTING_MUTUAL_METHODS = {
    "fmc_pc": "matched_random",
    "limb_shuffle_pc": "limb_shuffle",
}


@dataclass(frozen=True)
class SensitivitySetting:
    name: str
    probability: float
    consistency_weight: float
    reuse_center: bool = False


SENSITIVITY_SETTINGS = (
    SensitivitySetting("p025_l010", 0.25, 0.10),
    SensitivitySetting("p050_l005", 0.50, 0.05),
    SensitivitySetting("p050_l010", 0.50, 0.10, reuse_center=True),
    SensitivitySetting("p050_l020", 0.50, 0.20),
    SensitivitySetting("p075_l010", 0.75, 0.10),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def nested_float(payload: dict[str, Any], dotted: str) -> float:
    value: Any = payload
    for key in dotted.split("."):
        value = value[key]
    return float(value)


def evaluation_scenarios(
    ideal_manifest: str, self_manifest: str, mutual_manifest: str
) -> dict[str, tuple[str, tuple[str, ...]]]:
    return {
        "as_released_ideal": (ideal_manifest, AS_RELEASED_PARTICIPANTS),
        "as_released_self": (self_manifest, AS_RELEASED_PARTICIPANTS),
        "valid_signal_ideal": (ideal_manifest, VALID_SIGNAL_PARTICIPANTS),
        "valid_signal_self": (self_manifest, VALID_SIGNAL_PARTICIPANTS),
        "subject15_ideal": (ideal_manifest, MUTUAL_PARTICIPANT),
        "subject15_mutual": (mutual_manifest, MUTUAL_PARTICIPANT),
    }


def evaluate_checkpoint(
    checkpoint: Path,
    manifest: str,
    participants: tuple[str, ...],
    destination: Path,
    device: str,
) -> dict[str, Any]:
    metrics_path = destination / "metrics.json"
    if not metrics_path.exists():
        run(
            [
                sys.executable,
                str(ROOT / "scripts/evaluate_realdisp_checkpoint.py"),
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                manifest,
                "--output",
                str(destination),
                "--device",
                device,
                "--participants",
                *participants,
                "--num-workers",
                "4",
            ]
        )
    return read_json(metrics_path)


def train_and_evaluate_methods(
    *,
    phase_name: str,
    methods: dict[str, str],
    output_root: Path,
    profile: Path,
    ideal_manifest: str,
    self_manifest: str,
    mutual_manifest: str,
    device: str,
    stress_repeats: int,
) -> dict[str, Any]:
    phase_root = output_root / phase_name
    scenarios = evaluation_scenarios(ideal_manifest, self_manifest, mutual_manifest)
    total = len(methods) * len(SEEDS)
    position = 0
    runs: dict[str, Any] = {}
    for method, relative_config in methods.items():
        runs[method] = {}
        for seed in SEEDS:
            position += 1
            run_root = phase_root / method / f"seed{seed}"
            write_json(
                {
                    "phase": phase_name,
                    "position": position,
                    "total": total,
                    "method": method,
                    "seed": seed,
                    "state": "running",
                    "updated_utc": utc_now(),
                },
                phase_root / "queue_status.json",
            )
            validation_path = run_root / "validation_metrics.json"
            if not validation_path.exists():
                run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/run_realdisp_validation_diagnostic.py"),
                        "--config",
                        str(ROOT / relative_config),
                        "--output",
                        str(run_root),
                        "--device",
                        device,
                        "--seed",
                        str(seed),
                        "--stress-repeats",
                        str(stress_repeats),
                        "--placement-profile",
                        str(profile),
                    ]
                )
            checkpoint = run_root / "best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            run_payload: dict[str, Any] = {
                "validation": read_json(validation_path),
                "scenarios": {},
            }
            for scenario, (manifest, participants) in scenarios.items():
                run_payload["scenarios"][scenario] = evaluate_checkpoint(
                    checkpoint,
                    manifest,
                    participants,
                    run_root / scenario,
                    device,
                )
            write_json(
                {
                    "phase": phase_name,
                    "method": method,
                    "config": relative_config,
                    "config_sha256": sha256(ROOT / relative_config),
                    "profile_sha256": sha256(profile),
                    "seed": seed,
                    "test_selection_used": False,
                    "checkpoint_selection": "subjects11-12 ideal only",
                },
                run_root / "fmc_protocol.json",
            )
            runs[method][str(seed)] = run_payload
            write_json(runs, phase_root / "runs.partial.json")

    aggregate: dict[str, Any] = {}
    metrics = {
        "smoothed_macro_f1": "macro_f1",
        "smoothed_accuracy": "accuracy",
        "raw_macro_f1": "raw.macro_f1",
        "raw_accuracy": "raw.accuracy",
    }
    for method, seed_runs in runs.items():
        aggregate[method] = {}
        for scenario in scenarios:
            aggregate[method][scenario] = {
                label: metric_summary(
                    [
                        nested_float(seed_runs[str(seed)]["scenarios"][scenario], dotted)
                        for seed in SEEDS
                    ]
                )
                for label, dotted in metrics.items()
            }
    summary = {
        "method": f"FMC-PC {phase_name} completion",
        "protocol": {
            "supervised_train": list(TRAIN_PARTICIPANTS),
            "validation": list(VALIDATION_PARTICIPANTS),
            "as_released_test": list(AS_RELEASED_PARTICIPANTS),
            "post_hoc_valid_signal_test": list(VALID_SIGNAL_PARTICIPANTS),
            "mutual_unseen_participant": list(MUTUAL_PARTICIPANT),
            "seeds": list(SEEDS),
            "execution": "strictly sequential, resumable, fail-fast",
        },
        "aggregate": aggregate,
    }
    write_json(summary, phase_root / "summary.json")
    write_json(
        {
            "phase": phase_name,
            "position": total,
            "total": total,
            "state": "completed",
            "updated_utc": utc_now(),
        },
        phase_root / "queue_status.json",
    )
    (phase_root / "PHASE_COMPLETED").write_text(utc_now() + "\n", encoding="utf-8")
    return summary


def evaluate_existing_mutual(
    *,
    output_root: Path,
    strict_control_root: Path,
    ideal_manifest: str,
    mutual_manifest: str,
    device: str,
) -> dict[str, Any]:
    phase_root = output_root / "existing_mutual"
    runs: dict[str, Any] = {}
    for method, source_name in EXISTING_MUTUAL_METHODS.items():
        runs[method] = {}
        for seed in SEEDS:
            checkpoint = strict_control_root / source_name / f"seed{seed}" / "best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            runs[method][str(seed)] = {
                "ideal": evaluate_checkpoint(
                    checkpoint,
                    ideal_manifest,
                    MUTUAL_PARTICIPANT,
                    phase_root / method / f"seed{seed}" / "ideal",
                    device,
                ),
                "mutual": evaluate_checkpoint(
                    checkpoint,
                    mutual_manifest,
                    MUTUAL_PARTICIPANT,
                    phase_root / method / f"seed{seed}" / "mutual",
                    device,
                ),
            }
            write_json(runs, phase_root / "runs.partial.json")
    aggregate: dict[str, Any] = {}
    for method, seed_runs in runs.items():
        aggregate[method] = {}
        for scenario in ("ideal", "mutual"):
            aggregate[method][scenario] = {
                label: metric_summary(
                    [nested_float(seed_runs[str(seed)][scenario], dotted) for seed in SEEDS]
                )
                for label, dotted in {
                    "smoothed_macro_f1": "macro_f1",
                    "smoothed_accuracy": "accuracy",
                    "raw_macro_f1": "raw.macro_f1",
                    "raw_accuracy": "raw.accuracy",
                }.items()
            }
        aggregate[method]["mutual_minus_ideal_pp"] = {
            metric: 100.0
            * (
                aggregate[method]["mutual"][metric]["mean"]
                - aggregate[method]["ideal"][metric]["mean"]
            )
            for metric in ("smoothed_macro_f1", "raw_macro_f1")
        }
    summary = {
        "method": "Frozen FMC-PC negative-control mutual-displacement evaluation",
        "protocol": {
            "retraining": False,
            "unseen_test_participant": MUTUAL_PARTICIPANT[0],
            "excluded_seen_mutual_participants": ["subject2", "subject5"],
            "seeds": list(SEEDS),
        },
        "aggregate": aggregate,
    }
    write_json(summary, phase_root / "summary.json")
    (phase_root / "PHASE_COMPLETED").write_text(utc_now() + "\n", encoding="utf-8")
    return summary


def sensitivity_config(
    base_config: dict[str, Any], setting: SensitivitySetting
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config["training"]["quality_augmentation_probability"] = setting.probability
    config["loss"]["perturbation_prediction_consistency"] = (
        setting.consistency_weight
    )
    return config


def run_sensitivity(
    *,
    output_root: Path,
    profile: Path,
    strict_control_root: Path,
    self_manifest: str,
    device: str,
    stress_repeats: int,
) -> dict[str, Any]:
    phase_root = output_root / "validation_sensitivity"
    generated = phase_root / "generated_configs"
    base_relative = "configs/realdisp_pd_pc_matched_random.yaml"
    base = load_config(ROOT / base_relative)
    for setting in SENSITIVITY_SETTINGS:
        write_json(sensitivity_config(base, setting), generated / f"{setting.name}.json")

    total = len(SENSITIVITY_SETTINGS) * len(SEEDS)
    position = 0
    runs: dict[str, Any] = {}
    for setting in SENSITIVITY_SETTINGS:
        runs[setting.name] = {}
        for seed in SEEDS:
            position += 1
            run_root = phase_root / setting.name / f"seed{seed}"
            source = (
                strict_control_root / "matched_random" / f"seed{seed}"
                if setting.reuse_center
                else run_root
            )
            write_json(
                {
                    "phase": "validation_sensitivity",
                    "position": position,
                    "total": total,
                    "setting": setting.name,
                    "seed": seed,
                    "state": "running",
                    "center_checkpoint_reused": setting.reuse_center,
                    "final_test_accessed": False,
                    "updated_utc": utc_now(),
                },
                phase_root / "queue_status.json",
            )
            validation_path = source / "validation_metrics.json"
            if not validation_path.exists():
                run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/run_realdisp_validation_diagnostic.py"),
                        "--config",
                        str(generated / f"{setting.name}.json"),
                        "--output",
                        str(source),
                        "--device",
                        device,
                        "--seed",
                        str(seed),
                        "--stress-repeats",
                        str(stress_repeats),
                        "--placement-profile",
                        str(profile),
                    ]
                )
            checkpoint = source / "best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            self_metrics = evaluate_checkpoint(
                checkpoint,
                self_manifest,
                VALIDATION_PARTICIPANTS,
                run_root / "validation_self_subjects11_12",
                device,
            )
            validation = read_json(validation_path)
            runs[setting.name][str(seed)] = {
                "validation_ideal_macro_f1": float(validation["clean"]["macro_f1"]),
                "validation_rotation_stress_macro_f1": float(
                    validation["rotation_stress"]["macro_f1_mean"]
                ),
                "validation_self_macro_f1": float(self_metrics["macro_f1"]),
                "validation_ideal_accuracy": float(validation["clean"]["accuracy"]),
                "validation_self_accuracy": float(self_metrics["accuracy"]),
                "source_checkpoint": str(checkpoint),
            }
            write_json(
                {
                    "setting": asdict(setting),
                    "seed": seed,
                    "source_checkpoint": str(checkpoint),
                    "profile_sha256": sha256(profile),
                    "final_test_accessed": False,
                },
                run_root / "sensitivity_protocol.json",
            )
            write_json(runs, phase_root / "runs.partial.json")

    aggregate: dict[str, Any] = {}
    metric_names = (
        "validation_ideal_macro_f1",
        "validation_rotation_stress_macro_f1",
        "validation_self_macro_f1",
        "validation_ideal_accuracy",
        "validation_self_accuracy",
    )
    for setting in SENSITIVITY_SETTINGS:
        aggregate[setting.name] = {
            "setting": asdict(setting),
            **{
                metric: metric_summary(
                    [runs[setting.name][str(seed)][metric] for seed in SEEDS]
                )
                for metric in metric_names
            },
        }
    summary = {
        "method": "FMC-PC validation-only sensitivity",
        "protocol": {
            "purpose": "descriptive robustness; no final-test reselection",
            "supervised_train": list(TRAIN_PARTICIPANTS),
            "validation_ideal_and_self": list(VALIDATION_PARTICIPANTS),
            "final_test_accessed": False,
            "seeds": list(SEEDS),
            "frozen_center": {"probability": 0.50, "consistency_weight": 0.10},
        },
        "aggregate": aggregate,
    }
    write_json(summary, phase_root / "summary.json")
    write_json(
        {
            "phase": "validation_sensitivity",
            "position": total,
            "total": total,
            "state": "completed",
            "final_test_accessed": False,
            "updated_utc": utc_now(),
        },
        phase_root / "queue_status.json",
    )
    (phase_root / "PHASE_COMPLETED").write_text(utc_now() + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete frozen FMC-PC journal experiments.")
    parser.add_argument(
        "--phase",
        required=True,
        choices=("existing-mutual", "rcbd-ablation", "cross-backbone", "sensitivity"),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--placement-profile", required=True)
    parser.add_argument("--strict-control-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stress-repeats", type=int, default=3)
    parser.add_argument(
        "--ideal-manifest",
        default=(
            "/root/autodl-tmp/realdisp_project/data/realdisp_distal_33/"
            "manifest_ideal.csv"
        ),
    )
    parser.add_argument(
        "--self-manifest",
        default=(
            "/root/autodl-tmp/realdisp_project/data/realdisp_distal_33/"
            "manifest_self.csv"
        ),
    )
    parser.add_argument(
        "--mutual-manifest",
        default=(
            "/root/autodl-tmp/realdisp_project/data/"
            "realdisp_distal_33_mutual/manifest_mutual.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if args.stress_repeats < 1:
        raise ValueError("stress-repeats must be positive")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    profile = Path(args.placement_profile).resolve()
    strict_control_root = Path(args.strict_control_root).resolve()
    if sha256(profile) != FROZEN_PROFILE_SHA256:
        raise ValueError("Frozen placement profile SHA256 mismatch")
    for manifest in (args.ideal_manifest, args.self_manifest, args.mutual_manifest):
        if not Path(manifest).is_file():
            raise FileNotFoundError(manifest)
    write_json(
        {
            "phase": args.phase,
            "started_utc": utc_now(),
            "profile_sha256": sha256(profile),
            "placement_source_sha256": sha256(ROOT / "src/rcbd_har/placement.py"),
            "engine_source_sha256": sha256(ROOT / "src/rcbd_har/engine.py"),
            "runner_source_sha256": sha256(Path(__file__).resolve()),
        },
        output_root / f"{args.phase}_source_audit.json",
    )
    if args.phase == "existing-mutual":
        result = evaluate_existing_mutual(
            output_root=output_root,
            strict_control_root=strict_control_root,
            ideal_manifest=args.ideal_manifest,
            mutual_manifest=args.mutual_manifest,
            device=args.device,
        )
    elif args.phase == "rcbd-ablation":
        result = train_and_evaluate_methods(
            phase_name="rcbd_ablation",
            methods=RCBD_ABLATIONS,
            output_root=output_root,
            profile=profile,
            ideal_manifest=args.ideal_manifest,
            self_manifest=args.self_manifest,
            mutual_manifest=args.mutual_manifest,
            device=args.device,
            stress_repeats=args.stress_repeats,
        )
    elif args.phase == "cross-backbone":
        result = train_and_evaluate_methods(
            phase_name="cross_backbone",
            methods=CROSS_BACKBONE,
            output_root=output_root,
            profile=profile,
            ideal_manifest=args.ideal_manifest,
            self_manifest=args.self_manifest,
            mutual_manifest=args.mutual_manifest,
            device=args.device,
            stress_repeats=args.stress_repeats,
        )
    else:
        result = run_sensitivity(
            output_root=output_root,
            profile=profile,
            strict_control_root=strict_control_root,
            self_manifest=args.self_manifest,
            device=args.device,
            stress_repeats=args.stress_repeats,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
