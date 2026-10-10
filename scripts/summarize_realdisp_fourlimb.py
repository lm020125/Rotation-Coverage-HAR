from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from summarize_realdisp_wimusim import (
    SCENARIOS,
    SEEDS,
    VARIANTS,
    bootstrap_mean_ci,
    exact_sign_flip_pvalue,
    sample_std,
)


BASE_NAMES = {
    "rcbd": "realdisp_rcbd_33_seed",
    "ordered_diff": "realdisp_ordered_diff_33_seed",
    "absdiff": "realdisp_absdiff_33_seed",
    "concat": "realdisp_concat_33_seed",
}


def metric_path(root: Path, variant: str, seed: int, scenario: str) -> Path:
    return root / f"{BASE_NAMES[variant]}{seed}_{scenario}_participant_eval" / "metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize REALDISP four-limb ablations without seed pseudo-replication."
    )
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    loaded: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for scenario in SCENARIOS:
        loaded[scenario] = {}
        for variant in VARIANTS:
            loaded[scenario][variant] = {}
            for seed in SEEDS:
                path = metric_path(args.outputs, variant, seed, scenario)
                if not path.exists():
                    raise FileNotFoundError(path)
                loaded[scenario][variant][seed] = json.loads(path.read_text(encoding="utf-8"))

    summary: dict[str, Any] = {"seeds": list(SEEDS), "scenarios": {}}
    rows: list[dict[str, object]] = []
    participant_rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        scenario_summary: dict[str, Any] = {"variants": {}, "paired_vs_rcbd": {}}
        for variant in VARIANTS:
            metrics = loaded[scenario][variant]
            macro_f1 = [float(metrics[seed]["macro_f1"]) for seed in SEEDS]
            accuracy = [float(metrics[seed]["accuracy"]) for seed in SEEDS]
            swap_agreement = [
                float(metrics[seed]["swap_stress"]["prediction_agreement"]) for seed in SEEDS
            ]
            swap_f1_delta = [
                float(metrics[seed]["swap_stress"]["macro_f1_delta"]) for seed in SEEDS
            ]
            scenario_summary["variants"][variant] = {
                "macro_f1": {
                    "values": macro_f1,
                    "mean": mean(macro_f1),
                    "std": sample_std(macro_f1),
                },
                "accuracy": {
                    "values": accuracy,
                    "mean": mean(accuracy),
                    "std": sample_std(accuracy),
                },
                "swap_prediction_agreement": {
                    "values": swap_agreement,
                    "mean": mean(swap_agreement),
                    "std": sample_std(swap_agreement),
                },
                "swap_macro_f1_delta": {
                    "values": swap_f1_delta,
                    "mean": mean(swap_f1_delta),
                    "std": sample_std(swap_f1_delta),
                },
            }
            rows.append(
                {
                    "scenario": scenario,
                    "variant": variant,
                    "macro_f1_mean": mean(macro_f1),
                    "macro_f1_std": sample_std(macro_f1),
                    "accuracy_mean": mean(accuracy),
                    "accuracy_std": sample_std(accuracy),
                    "swap_prediction_agreement_mean": mean(swap_agreement),
                    "swap_macro_f1_delta_mean": mean(swap_f1_delta),
                }
            )

        rcbd_metrics = loaded[scenario]["rcbd"]
        participants = sorted(rcbd_metrics[SEEDS[0]]["per_participant"])
        for variant in VARIANTS[1:]:
            participant_deltas: list[float] = []
            for participant in participants:
                rcbd_value = mean(
                    float(rcbd_metrics[seed]["per_participant"][participant]["macro_f1"])
                    for seed in SEEDS
                )
                comparator_value = mean(
                    float(
                        loaded[scenario][variant][seed]["per_participant"][participant][
                            "macro_f1"
                        ]
                    )
                    for seed in SEEDS
                )
                delta = rcbd_value - comparator_value
                participant_deltas.append(delta)
                participant_rows.append(
                    {
                        "scenario": scenario,
                        "comparator": variant,
                        "participant": participant,
                        "rcbd_macro_f1": rcbd_value,
                        "comparator_macro_f1": comparator_value,
                        "delta": delta,
                    }
                )
            scenario_summary["paired_vs_rcbd"][variant] = {
                "unit": "participant after averaging across seeds",
                "participants": participants,
                "macro_f1_deltas": participant_deltas,
                "mean_delta": mean(participant_deltas),
                "bootstrap_95_ci": bootstrap_mean_ci(participant_deltas),
                "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(participant_deltas),
            }
        summary["scenarios"][scenario] = scenario_summary

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for filename, data in (("summary.csv", rows), ("participant_deltas.csv", participant_rows)):
        with (args.output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()
