"""Frozen mechanism-control ablation on REALDISP (C1-C4, +/-PC, three seeds).

Pre-registered cells (EXP-3/EXP-2 of the 2026-08-01 editorial roadmap):

- ``c1_uniform180_no_pc`` / ``c1_uniform180_pc``: generic one-side rotation with
  support U(0, 180 deg); isolates the SUPPORT RANGE factor against the frozen
  U(0, 90 deg) cells of the review 2x2.
- ``c2_beta_no_pc`` / ``c2_beta_pc``: amplitude source replaced by the stratified
  Beta quantile pool matched to the empirical support/mean/SD; isolates the
  EMPIRICAL FREQUENCY factor from its first moments.
- ``c3_clip_no_pc`` / ``c3_clip_pc``: empirical pool with the five angles above
  90 deg clipped to 90 deg; tail-robustness of the empirical source.
- ``c4_gt_no_pc`` / ``c4_gt_pc``: empirical amplitude with the gravity-matched
  generator (exact gravity tilt, the generator used by the WEAR adapter);
  generator bridge between the REALDISP random-axis implementation and WEAR.

All cells freeze augmentation probability 0.5, matched one-side-per-pair
application, class-weighted supervision, identical participants/splits/early
stopping/smoothing/seeds (256, 3407, 114514), and rotation-only profile use
(gain/noise disabled) except C1 which uses no profile at all.  Profile files are
SHA-frozen; any mismatch aborts before training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (256, 3407, 114514)
VALID_SIGNAL = tuple(f"subject{value}" for value in range(14, 18))
AS_RELEASED = tuple(f"subject{value}" for value in range(13, 18))

FROZEN_PROFILE_SHA256 = (
    "b05eedb46aa01cff0c2331b09bc3ca4911821a0ef7c8291c0807dd19c555b2b7"
)
BETA_PROFILE_SHA256 = (
    "89de4e69d5750ad1fdf37c044644248c55b31a65c265819af5619224bb1c14d0"
)
CLIP_PROFILE_SHA256 = (
    "993d9ce11661af0810cc8cf77d499d034d725bdbb03b84dd86e39f919e76ad66"
)
PROFILE_DIR = ROOT / "profiles_mechanism_controls"

METHODS: dict[str, dict[str, Any]] = {
    "c1_uniform180_no_pc": {
        "amplitude_source": "uniform_0_180_degrees",
        "generator": "random_axis",
        "prediction_consistency": False,
        "config": "configs/realdisp_uniform180_no_pc.yaml",
        "profile": None,
        "profile_sha256": None,
    },
    "c1_uniform180_pc": {
        "amplitude_source": "uniform_0_180_degrees",
        "generator": "random_axis",
        "prediction_consistency": True,
        "config": "configs/realdisp_uniform180_pc.yaml",
        "profile": None,
        "profile_sha256": None,
    },
    "c2_beta_no_pc": {
        "amplitude_source": "beta_moment_matched_pool",
        "generator": "random_axis",
        "prediction_consistency": False,
        "config": "configs/realdisp_fmc_angle_no_pc.yaml",
        "profile": "placement_profiles_beta_moment_matched.json",
        "profile_sha256": BETA_PROFILE_SHA256,
    },
    "c2_beta_pc": {
        "amplitude_source": "beta_moment_matched_pool",
        "generator": "random_axis",
        "prediction_consistency": True,
        "config": "configs/realdisp_fmc_angle_pc.yaml",
        "profile": "placement_profiles_beta_moment_matched.json",
        "profile_sha256": BETA_PROFILE_SHA256,
    },
    "c3_clip_no_pc": {
        "amplitude_source": "empirical_clip90_pool",
        "generator": "random_axis",
        "prediction_consistency": False,
        "config": "configs/realdisp_fmc_angle_no_pc.yaml",
        "profile": "placement_profiles_clip90.json",
        "profile_sha256": CLIP_PROFILE_SHA256,
    },
    "c3_clip_pc": {
        "amplitude_source": "empirical_clip90_pool",
        "generator": "random_axis",
        "prediction_consistency": True,
        "config": "configs/realdisp_fmc_angle_pc.yaml",
        "profile": "placement_profiles_clip90.json",
        "profile_sha256": CLIP_PROFILE_SHA256,
    },
    "c4_gt_no_pc": {
        "amplitude_source": "training_subject_ideal_self_empirical_angle",
        "generator": "gravity_matched",
        "prediction_consistency": False,
        "config": "configs/realdisp_fmc_gt_no_pc.yaml",
        "profile": "placement_profiles_subjects1_10.json",
        "profile_sha256": FROZEN_PROFILE_SHA256,
    },
    "c4_gt_pc": {
        "amplitude_source": "training_subject_ideal_self_empirical_angle",
        "generator": "gravity_matched",
        "prediction_consistency": True,
        "config": "configs/realdisp_fmc_gt_pc.yaml",
        "profile": "placement_profiles_subjects1_10.json",
        "profile_sha256": FROZEN_PROFILE_SHA256,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def metric_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
    }


def nested_float(payload: dict[str, Any], dotted: str) -> float:
    value: Any = payload
    for key in dotted.split("."):
        value = value[key]
    return float(value)


def validate_config(method: str, config: dict[str, Any], *, checkpoint: bool = False) -> None:
    expected = METHODS[method]
    training = config["training"]
    loss = config["loss"]
    if float(training["quality_augmentation_probability"]) != 0.5:
        raise ValueError(f"{method}: augmentation probability is not frozen at 0.5")
    if float(training.get("quality_augmentation_classification_mix", -1.0)) != 0.5:
        raise ValueError(f"{method}: classification mix is not frozen at 0.5")
    actual_pc = float(loss.get("perturbation_prediction_consistency", 0.0))
    expected_pc = 0.1 if expected["prediction_consistency"] else 0.0
    if actual_pc != expected_pc:
        raise ValueError(f"{method}: expected PC={expected_pc}, found {actual_pc}")
    if expected["generator"] == "random_axis" and expected["profile"] is None:
        if "quality_augmentation_profile_path" in training:
            raise ValueError(f"{method}: uniform cell must not carry a profile path")
        if float(training.get("quality_augmentation_max_degrees", -1.0)) != 180.0:
            raise ValueError(f"{method}: uniform cell must use 0-180 degrees")
        return
    if not checkpoint and "quality_augmentation_profile_path" in training:
        raise ValueError(
            f"{method}: profile path must be injected at runtime, not baked into config"
        )
    required = {
        "quality_augmentation_profile_rotation": True,
        "quality_augmentation_profile_gain": False,
        "quality_augmentation_profile_noise": False,
    }
    for key, value in required.items():
        if training.get(key) != value:
            raise ValueError(f"{method}: {key} must equal {value!r}")
    expected_sampling = (
        "gravity_matched_random"
        if expected["generator"] == "gravity_matched"
        else "matched_random"
    )
    if training.get("quality_augmentation_profile_sampling") != expected_sampling:
        raise ValueError(
            f"{method}: sampling must equal {expected_sampling!r}, found "
            f"{training.get('quality_augmentation_profile_sampling')!r}"
        )


def load_checkpoint_config(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state["config"]


def evaluate(
    checkpoint: Path,
    manifest: Path,
    participants: tuple[str, ...],
    output: Path,
    device: str,
) -> dict[str, Any]:
    metrics_path = output / "metrics.json"
    if not metrics_path.exists():
        run(
            [
                sys.executable,
                str(ROOT / "scripts/evaluate_realdisp_checkpoint.py"),
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--device",
                device,
                "--participants",
                *participants,
                "--num-workers",
                "4",
            ]
        )
    return read_json(metrics_path)


def aggregate(runs: dict[str, Any]) -> dict[str, Any]:
    scenarios = ("valid_signal_ideal", "valid_signal_self", "as_released_self")
    metric_paths = {
        "macro_f1": "macro_f1",
        "accuracy": "accuracy",
        "raw_macro_f1": "raw.macro_f1",
        "raw_accuracy": "raw.accuracy",
    }
    methods: dict[str, Any] = {}
    for method, method_runs in runs.items():
        methods[method] = {
            "design": {
                "amplitude_source": METHODS[method]["amplitude_source"],
                "generator": METHODS[method]["generator"],
                "prediction_consistency": METHODS[method]["prediction_consistency"],
                "profile_sha256": METHODS[method]["profile_sha256"],
            },
            "scenarios": {},
            "valid_signal_self_per_participant": {},
            "valid_signal_self_per_class": {},
        }
        for scenario in scenarios:
            methods[method]["scenarios"][scenario] = {
                label: metric_summary(
                    [
                        nested_float(method_runs[str(seed)][scenario], path)
                        for seed in SEEDS
                    ]
                )
                for label, path in metric_paths.items()
            }
        for participant in VALID_SIGNAL:
            methods[method]["valid_signal_self_per_participant"][participant] = {
                label: metric_summary(
                    [
                        float(
                            method_runs[str(seed)]["valid_signal_self"][
                                "per_participant"
                            ][participant][key]
                        )
                        for seed in SEEDS
                    ]
                )
                for label, key in {
                    "macro_f1": "macro_f1",
                    "accuracy": "accuracy",
                }.items()
            }
        for class_id in range(33):
            key = str(class_id)
            methods[method]["valid_signal_self_per_class"][key] = {
                "f1": metric_summary(
                    [
                        float(
                            method_runs[str(seed)]["valid_signal_self"]["per_class"][
                                key
                            ]["f1-score"]
                        )
                        for seed in SEEDS
                    ]
                ),
                "support": int(
                    method_runs[str(SEEDS[0])]["valid_signal_self"]["per_class"][
                        key
                    ]["support"]
                ),
            }
    return {
        "method": "REALDISP mechanism controls (C1-C4 x PC, frozen)",
        "protocol": {
            "cells": {
                name: {
                    "amplitude_source": spec["amplitude_source"],
                    "generator": spec["generator"],
                    "prediction_consistency": spec["prediction_consistency"],
                }
                for name, spec in METHODS.items()
            },
            "seeds": list(SEEDS),
            "valid_signal": list(VALID_SIGNAL),
            "as_released": list(AS_RELEASED),
            "checkpoint_selection": "subjects11-12 Ideal only",
            "test_based_method_selection": False,
            "same_model_predictions_for_both_test_scopes": True,
            "frozen_profile_sha256": FROZEN_PROFILE_SHA256,
            "beta_profile_sha256": BETA_PROFILE_SHA256,
            "clip_profile_sha256": CLIP_PROFILE_SHA256,
            "pre_registered": True,
        },
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--ideal-manifest", required=True, type=Path)
    parser.add_argument("--self-manifest", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stress-repeats", type=int, default=3)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for path in (args.ideal_manifest, args.self_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    profile_paths: dict[str, Path | None] = {}
    for spec in METHODS.values():
        name = spec["profile"]
        if name is None:
            profile_paths[name] = None
            continue
        path = PROFILE_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path)
        if digest != spec["profile_sha256"]:
            raise ValueError(
                f"Profile SHA mismatch for {name}: expected {spec['profile_sha256']}, "
                f"found {digest}"
            )
        profile_paths[name] = path

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runs: dict[str, Any] = {}
    total = len(METHODS) * len(SEEDS)
    position = 0

    for method, specification in METHODS.items():
        config_path = ROOT / str(specification["config"])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        validate_config(method, config)
        profile_path = profile_paths[specification["profile"]]
        runs[method] = {}
        for seed in SEEDS:
            position += 1
            write_json(
                {
                    "state": "running",
                    "position": position,
                    "total": total,
                    "method": method,
                    "seed": seed,
                    "updated_utc": utc_now(),
                },
                output_root / "queue_status.json",
            )
            run_root = output_root / method / f"seed{seed}"
            validation = run_root / "validation_metrics.json"
            if not validation.exists():
                command = [
                    sys.executable,
                    str(ROOT / "scripts/run_realdisp_validation_diagnostic.py"),
                    "--config",
                    str(config_path),
                    "--output",
                    str(run_root),
                    "--device",
                    args.device,
                    "--seed",
                    str(seed),
                    "--stress-repeats",
                    str(args.stress_repeats),
                ]
                if profile_path is not None:
                    command += ["--placement-profile", str(profile_path)]
                run(command)
            checkpoint = run_root / "best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            checkpoint_config = load_checkpoint_config(checkpoint)
            validate_config(method, checkpoint_config, checkpoint=True)
            if profile_path is not None:
                recorded = Path(
                    str(checkpoint_config["training"]["quality_augmentation_profile_path"])
                )
                if sha256(recorded) != specification["profile_sha256"]:
                    raise ValueError(
                        f"{method}: checkpoint profile SHA does not match the frozen value"
                    )
            scenario_metrics = {
                "valid_signal_ideal": evaluate(
                    checkpoint,
                    args.ideal_manifest,
                    VALID_SIGNAL,
                    run_root / "valid_signal_ideal",
                    args.device,
                ),
                "valid_signal_self": evaluate(
                    checkpoint,
                    args.self_manifest,
                    VALID_SIGNAL,
                    run_root / "valid_signal_self",
                    args.device,
                ),
                "as_released_self": evaluate(
                    checkpoint,
                    args.self_manifest,
                    AS_RELEASED,
                    run_root / "as_released_self",
                    args.device,
                ),
            }
            write_json(
                {
                    "method": method,
                    "seed": seed,
                    "design": {
                        "amplitude_source": specification["amplitude_source"],
                        "generator": specification["generator"],
                        "prediction_consistency": specification[
                            "prediction_consistency"
                        ],
                    },
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "config": str(config_path),
                    "config_sha256": sha256(config_path),
                    "placement_profile": (
                        str(profile_path) if profile_path is not None else None
                    ),
                    "placement_profile_sha256": specification["profile_sha256"],
                    "checkpoint_selection": "subjects11-12 Ideal only",
                    "test_based_method_selection": False,
                },
                run_root / "review_protocol.json",
            )
            runs[method][str(seed)] = scenario_metrics
            write_json(runs, output_root / "runs.partial.json")

    summary = aggregate(runs)
    write_json(summary, output_root / "summary.json")
    write_json(
        {
            "state": "completed",
            "position": total,
            "total": total,
            "updated_utc": utc_now(),
        },
        output_root / "queue_status.json",
    )
    (output_root / "COMPLETED").write_text(utc_now() + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
