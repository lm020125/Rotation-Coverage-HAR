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


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("pd_rcbd", "pd_prediction_consistency")
SCENARIOS = ("ideal", "self")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_final(
    methods: dict[str, dict[str, dict[str, dict[str, Any]]]]
) -> dict[str, Any]:
    seeds = sorted(methods["pd_rcbd"], key=int)
    aggregate: dict[str, Any] = {scenario: {} for scenario in SCENARIOS}
    for scenario in SCENARIOS:
        for method in METHODS:
            results = methods[method]
            aggregate[scenario][method] = {}
            for metric in ("macro_f1", "accuracy"):
                values = np.asarray(
                    [results[seed][scenario][metric] for seed in seeds], dtype=float
                )
                aggregate[scenario][method][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "values": values.tolist(),
                }
        deltas = {
            seed: {
                metric: 100.0
                * (
                    methods["pd_prediction_consistency"][seed][scenario][metric]
                    - methods["pd_rcbd"][seed][scenario][metric]
                )
                for metric in ("macro_f1", "accuracy")
            }
            for seed in seeds
        }
        aggregate[scenario]["pd_pc_minus_pd_rcbd_pp"] = {
            "per_seed": deltas,
            "macro_f1_mean": float(
                np.mean([value["macro_f1"] for value in deltas.values()])
            ),
            "accuracy_mean": float(
                np.mean([value["accuracy"] for value in deltas.values()])
            ),
        }
        participants = sorted(
            methods["pd_rcbd"][seeds[0]][scenario]["per_participant_macro_f1"]
        )
        aggregate[scenario]["participant_mean_macro_f1"] = {
            participant: {
                method: float(
                    np.mean(
                        [
                            methods[method][seed][scenario][
                                "per_participant_macro_f1"
                            ][participant]
                            for seed in seeds
                        ]
                    )
                )
                for method in METHODS
            }
            for participant in participants
        }
        for participant, values in aggregate[scenario][
            "participant_mean_macro_f1"
        ].items():
            values["pd_pc_minus_pd_rcbd_pp"] = 100.0 * (
                values["pd_prediction_consistency"] - values["pd_rcbd"]
            )
    return aggregate


def checkpoint_path(
    seed: int, method: str, multiseed_root: Path, seed3407_root: Path
) -> Path:
    return (
        seed3407_root / method / "best.pt"
        if seed == 3407
        else multiseed_root / f"seed_{seed}" / method / "best.pt"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot frozen PD-RCBD/PD-PC evaluation on REALDISP subjects13-17."
        )
    )
    parser.add_argument("--multiseed-root", required=True)
    parser.add_argument("--seed3407-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[256, 3407, 114514])
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
    args = parser.parse_args()

    multiseed_root = Path(args.multiseed_root).resolve()
    seed3407_root = Path(args.seed3407_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    participants = [f"subject{value}" for value in range(13, 18)]
    manifests = {"ideal": args.ideal_manifest, "self": args.self_manifest}
    profile_hash = read_json(multiseed_root / "summary.json")["profile"]["sha256"]
    methods: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        method: {} for method in METHODS
    }
    audit: dict[str, Any] = {}
    summary: dict[str, Any] = {
        "method": "Frozen PD-PC final REALDISP evaluation",
        "protocol": {
            "participants": participants,
            "scenarios": list(SCENARIOS),
            "seeds": args.seeds,
            "retraining": False,
            "method_or_profile_modified_after_validation": False,
            "final_test_accessed": True,
            "access_started_utc": datetime.now(timezone.utc).isoformat(),
        },
        "frozen_profile_sha256": profile_hash,
        "methods": methods,
        "checkpoint_audit": audit,
    }
    for seed in args.seeds:
        for method in METHODS:
            checkpoint = checkpoint_path(seed, method, multiseed_root, seed3407_root)
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            configured_profile = Path(
                state["config"]["training"]["quality_augmentation_profile_path"]
            ).resolve()
            if sha256(configured_profile) != profile_hash:
                raise RuntimeError(
                    f"Checkpoint {checkpoint} does not reference the frozen profile"
                )
            audit[f"{method}/seed_{seed}"] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "checkpoint_epoch": int(state.get("epoch", -1)),
                "profile": str(configured_profile),
                "profile_sha256": profile_hash,
            }
            methods[method][str(seed)] = {}
            for scenario in SCENARIOS:
                output = output_root / f"seed_{seed}" / method / scenario
                metrics_path = output / "metrics.json"
                if not metrics_path.exists():
                    run(
                        [
                            sys.executable,
                            str(ROOT / "scripts" / "evaluate_realdisp_checkpoint.py"),
                            "--checkpoint",
                            str(checkpoint),
                            "--manifest",
                            manifests[scenario],
                            "--output",
                            str(output),
                            "--device",
                            args.device,
                            "--participants",
                            *participants,
                            "--num-workers",
                            "4",
                        ]
                    )
                metrics = read_json(metrics_path)
                methods[method][str(seed)][scenario] = {
                    "macro_f1": float(metrics["macro_f1"]),
                    "accuracy": float(metrics["accuracy"]),
                    "weighted_f1": float(metrics["weighted_f1"]),
                    "per_participant_macro_f1": {
                        participant: float(values["macro_f1"])
                        for participant, values in metrics["per_participant"].items()
                    },
                }
                with (output_root / "summary.partial.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(summary, handle, ensure_ascii=False, indent=2)
    summary["aggregate"] = summarize_final(methods)
    summary["protocol"]["access_completed_utc"] = datetime.now(timezone.utc).isoformat()
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
