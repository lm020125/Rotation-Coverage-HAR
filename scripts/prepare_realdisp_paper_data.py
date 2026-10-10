"""Prepare tidy, auditable REALDISP result tables for paper figures and tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


SEEDS = (256, 3407, 114514)
METRICS = (
    "validation_ideal_macro_f1",
    "validation_rotation_stress_macro_f1",
    "final_ideal_macro_f1",
    "final_self_macro_f1",
    "final_ideal_accuracy",
    "final_self_accuracy",
)
ABLATION_LABELS = {
    "clean_rcbd": "Clean RCBD",
    "random_rotation": "Random rotation",
    "pd_rotation_only": "PD rotation",
    "pd_rotation_gain": "PD rotation + gain",
    "pd_full_no_pc": "PD full (no PC)",
    "random_rotation_pc": "Random rotation + PC",
    "pd_pc": "PD-PC",
}
DCL_LABELS = {
    "dcl_clean": "Clean",
    "dcl_random_rotation": "Random rotation",
    "dcl_pd_pc": "PD-PC",
}
MAIN_METHODS = ("clean_rcbd", "random_rotation", "pd_pc")
CONDITION_LABELS = {
    "clean_rcbd": "Clean",
    "random_rotation": "Random rotation",
    "pd_pc": "PD-PC",
    "dcl_clean": "Clean",
    "dcl_random_rotation": "Random rotation",
    "dcl_pd_pc": "PD-PC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--realdisp-root",
        type=Path,
        required=True,
        help="REALDISP archive directory containing outputs/ and logs/.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary_frames(
    summary: dict[str, Any],
    labels: dict[str, str],
    model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for order, (method, label) in enumerate(labels.items()):
        payload = summary["methods"][method]
        for run in payload["runs"]:
            row = {
                "model": model,
                "method": method,
                "method_label": label,
                "method_order": order,
                "seed": int(run["seed"]),
            }
            row.update({metric: float(run[metric]) * 100 for metric in METRICS})
            run_rows.append(row)
        summary_row = {
            "model": model,
            "method": method,
            "method_label": label,
            "method_order": order,
            "n_seeds": len(payload["runs"]),
        }
        for metric in METRICS:
            summary_row[f"{metric}_mean"] = float(payload[metric]["mean"]) * 100
            summary_row[f"{metric}_std"] = float(payload[metric]["std"]) * 100
        summary_rows.append(summary_row)
    return pd.DataFrame(run_rows), pd.DataFrame(summary_rows)


def sensitivity_frames(summary: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    metrics = (
        "validation_ideal_macro_f1",
        "validation_rotation_stress_macro_f1",
        "validation_self_macro_f1",
        "validation_ideal_accuracy",
        "validation_self_accuracy",
    )
    for order, (name, payload) in enumerate(summary["settings"].items()):
        setting = payload["setting"]
        shared = {
            "setting": name,
            "setting_order": order,
            "augmentation_probability": float(setting["augmentation_probability"]),
            "consistency_weight": float(setting["consistency_weight"]),
            "is_frozen_center": bool(setting["reuse_frozen_center"]),
        }
        for run in payload["runs"]:
            row = dict(shared)
            row["seed"] = int(run["seed"])
            row.update({metric: float(run[metric]) * 100 for metric in metrics})
            run_rows.append(row)
        summary_row = dict(shared)
        summary_row["n_seeds"] = len(payload["runs"])
        for metric in metrics:
            summary_row[f"{metric}_mean"] = float(payload[metric]["mean"]) * 100
            summary_row[f"{metric}_std"] = float(payload[metric]["std"]) * 100
        summary_rows.append(summary_row)
    return pd.DataFrame(run_rows), pd.DataFrame(summary_rows)


def participant_long(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pd.read_csv(source)
    main = wide.loc[
        (wide["scenario"] == "final_self_subjects13_17")
        & wide["method"].isin(MAIN_METHODS)
    ].copy()
    main["method_label"] = main["method"].map(ABLATION_LABELS)
    main["method_order"] = main["method"].map({key: i for i, key in enumerate(MAIN_METHODS)})
    main["data_status"] = main["participant"].map(
        lambda participant: "all-zero sensor record" if participant == "subject13" else "valid"
    )
    long = main.melt(
        id_vars=[
            "scenario",
            "participant",
            "data_status",
            "method",
            "method_label",
            "method_order",
        ],
        value_vars=[f"seed_{seed}" for seed in SEEDS],
        var_name="seed",
        value_name="macro_f1",
    )
    long["seed"] = long["seed"].str.removeprefix("seed_").astype(int)
    long["macro_f1"] *= 100
    for column in [*(f"seed_{seed}" for seed in SEEDS), "mean", "std"]:
        main[column] *= 100
    return long.sort_values(["participant", "method_order", "seed"]), main


def build_catalog(sources: list[Path], outputs: list[Path], protocol: dict[str, Any]) -> dict:
    return {
        "purpose": "frozen REALDISP paper figure and table data",
        "protocol": protocol,
        "source_files": [
            {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sources
        ],
        "generated_files": [
            {"name": path.name, "rows": int(len(pd.read_csv(path)))} for path in outputs
        ],
    }


def main() -> None:
    args = parse_args()
    realdisp_root = args.realdisp_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    outputs_root = realdisp_root / "outputs"

    ablation_path = (
        outputs_root / "realdisp_pd_pc_component_ablation_20260718" / "summary.json"
    )
    dcl_path = (
        outputs_root / "realdisp_deepconvlstm_generalization_20260719" / "summary.json"
    )
    sensitivity_path = (
        outputs_root / "realdisp_pd_pc_sensitivity_20260719" / "summary.json"
    )
    participant_path = (
        outputs_root
        / "realdisp_pd_pc_statistical_audit_20260719"
        / "participant_metrics.csv"
    )
    class_path = (
        outputs_root / "realdisp_pd_pc_statistical_audit_20260719" / "class_metrics.csv"
    )
    sources = [ablation_path, dcl_path, sensitivity_path, participant_path, class_path]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required frozen result files: {missing}")

    ablation = read_json(ablation_path)
    dcl = read_json(dcl_path)
    sensitivity = read_json(sensitivity_path)
    if sensitivity["protocol"].get("final_test_accessed") is not False:
        raise ValueError("Sensitivity protocol does not confirm final_test_accessed=false")

    ablation_runs, ablation_summary = summary_frames(
        ablation, ABLATION_LABELS, model="RCBD"
    )
    dcl_runs, dcl_summary = summary_frames(dcl, DCL_LABELS, model="DeepConvLSTM")
    sensitivity_runs, sensitivity_summary = sensitivity_frames(sensitivity)
    participant_runs, participant_summary = participant_long(participant_path)
    class_metrics = pd.read_csv(class_path)
    class_main = class_metrics.loc[
        (class_metrics["scenario"] == "final_self_subjects13_17")
        & class_metrics["method"].isin(MAIN_METHODS)
    ].copy()
    class_main["method_label"] = class_main["method"].map(ABLATION_LABELS)
    class_main["mean_f1"] *= 100
    class_main["std_f1"] *= 100

    main_comparison = pd.concat(
        [
            ablation_runs.loc[ablation_runs["method"].isin(MAIN_METHODS)],
            dcl_runs,
        ],
        ignore_index=True,
    )
    main_comparison["condition"] = main_comparison["method"].map(CONDITION_LABELS)
    main_comparison["condition_order"] = main_comparison["condition"].map(
        {"Clean": 0, "Random rotation": 1, "PD-PC": 2}
    )

    frames = {
        "ablation_runs.csv": ablation_runs,
        "ablation_summary.csv": ablation_summary,
        "backbone_runs.csv": main_comparison,
        "deepconvlstm_summary.csv": dcl_summary,
        "sensitivity_runs.csv": sensitivity_runs,
        "sensitivity_summary.csv": sensitivity_summary,
        "participant_self_runs.csv": participant_runs,
        "participant_self_summary.csv": participant_summary,
        "class_self_main.csv": class_main,
    }
    written: list[Path] = []
    for filename, frame in frames.items():
        target = output / filename
        frame.to_csv(target, index=False, float_format="%.8f")
        written.append(target)

    catalog = build_catalog(
        sources,
        written,
        protocol={
            "supervised_train": "subjects1-10",
            "validation": "subjects11-12",
            "frozen_final_test": "subjects13-17",
            "seeds": list(SEEDS),
            "sensitivity_is_post_freeze_validation_only": True,
            "subject13_self_status": (
                "all four limb arrays are all-zero; retained in the frozen "
                "as-released benchmark and excluded only from the post hoc "
                "valid-signal conditional analysis"
            ),
            "paired_headline_estimands": {
                "as_released": "subjects13-17; frozen benchmark",
                "valid_signal_conditional": (
                    "subjects14-17; post hoc label-free raw-input rule"
                ),
            },
            "quality_rule_uses_labels_or_predictions": False,
            "quality_analysis_retrains_models": False,
        },
    )
    with (output / "paper_result_catalog.json").open("w", encoding="utf-8") as handle:
        json.dump(catalog, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps({"output": str(output), "files": len(written) + 1}, indent=2))


if __name__ == "__main__":
    main()
