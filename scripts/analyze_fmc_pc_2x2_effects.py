from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CELL_NAMES = (
    "uniform_no_pc",
    "uniform_pc",
    "empirical_no_pc",
    "empirical_pc",
)
DEFAULT_METRICS = (
    "macro_f1",
    "accuracy",
    "raw_macro_f1",
    "raw_accuracy",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def paired_cell_values(
    summary: dict[str, Any],
    scenario: str,
    metric: str,
) -> tuple[list[int], np.ndarray]:
    seeds = [int(seed) for seed in summary["protocol"]["seeds"]]
    methods = summary["methods"]
    rows: list[np.ndarray] = []
    for method in CELL_NAMES:
        values = methods[method]["scenarios"][scenario][metric]["values"]
        row = np.asarray(values, dtype=np.float64)
        if row.shape != (len(seeds),):
            raise ValueError(
                f"{method}/{scenario}/{metric} has {row.size} values; "
                f"expected one value for each of {len(seeds)} seeds"
            )
        if not np.isfinite(row).all():
            raise ValueError(f"{method}/{scenario}/{metric} contains non-finite values")
        rows.append(row)
    return seeds, np.stack(rows, axis=1)


def factorial_effects(cells: np.ndarray) -> dict[str, np.ndarray]:
    """Compute paired 2x2 effects from [U0, U1, E0, E1] per seed."""
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("cells must have shape [number_of_seeds, 4]")
    uniform_no_pc, uniform_pc, empirical_no_pc, empirical_pc = cells.T
    amplitude = 0.5 * (
        (empirical_no_pc - uniform_no_pc) + (empirical_pc - uniform_pc)
    )
    prediction_consistency = 0.5 * (
        (uniform_pc - uniform_no_pc) + (empirical_pc - empirical_no_pc)
    )
    interaction = (empirical_pc - empirical_no_pc) - (
        uniform_pc - uniform_no_pc
    )
    return {
        "empirical_amplitude_main_effect": amplitude,
        "prediction_consistency_main_effect": prediction_consistency,
        "amplitude_x_pc_interaction": interaction,
    }


def paired_bootstrap_ci(
    values: np.ndarray,
    *,
    repeats: int,
    random_seed: int,
) -> tuple[float, float]:
    if values.ndim != 1 or values.size < 1:
        raise ValueError("values must be a non-empty one-dimensional array")
    if repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, values.size, size=(repeats, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lower), float(upper)


def summarize_effect(
    values: np.ndarray,
    *,
    repeats: int,
    random_seed: int,
) -> dict[str, Any]:
    lower, upper = paired_bootstrap_ci(
        values,
        repeats=repeats,
        random_seed=random_seed,
    )
    return {
        "values": values.tolist(),
        "values_percentage_points": (100.0 * values).tolist(),
        "mean": float(values.mean()),
        "mean_percentage_points": float(100.0 * values.mean()),
        "std_population": float(values.std(ddof=0)),
        "std_population_percentage_points": float(100.0 * values.std(ddof=0)),
        "paired_bootstrap_95_percentile_ci": [lower, upper],
        "paired_bootstrap_95_percentile_ci_percentage_points": [
            100.0 * lower,
            100.0 * upper,
        ],
    }


def analyze_summary(
    summary: dict[str, Any],
    *,
    scenario: str,
    metrics: tuple[str, ...],
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "analysis": "paired FMC-PC 2x2 factorial effects",
        "scenario": scenario,
        "cell_order": list(CELL_NAMES),
        "effect_definitions": {
            "empirical_amplitude_main_effect": (
                "0.5 * [(empirical_no_pc - uniform_no_pc) + "
                "(empirical_pc - uniform_pc)]"
            ),
            "prediction_consistency_main_effect": (
                "0.5 * [(uniform_pc - uniform_no_pc) + "
                "(empirical_pc - empirical_no_pc)]"
            ),
            "amplitude_x_pc_interaction": (
                "(empirical_pc - empirical_no_pc) - "
                "(uniform_pc - uniform_no_pc)"
            ),
        },
        "bootstrap": {
            "method": "paired seed bootstrap with joint resampling of 2x2 cells",
            "repeats": int(bootstrap_repeats),
            "random_seed": int(bootstrap_seed),
            "interval": "percentile 95%",
        },
        "interpretation_guardrail": {
            "status": "descriptive_only",
            "reason": (
                "Only three training seeds are available. The paired bootstrap "
                "intervals describe seed sensitivity and are not confirmatory "
                "population-level confidence intervals."
            ),
            "p_values_computed": False,
            "significance_claim_authorized": False,
        },
        "metrics": {},
    }
    reference_seeds: list[int] | None = None
    for metric_index, metric in enumerate(metrics):
        seeds, cells = paired_cell_values(summary, scenario, metric)
        if reference_seeds is None:
            reference_seeds = seeds
            output["seeds"] = seeds
            output["n_seeds"] = len(seeds)
        elif seeds != reference_seeds:
            raise ValueError("Metric seed order is inconsistent")
        effects = factorial_effects(cells)
        output["metrics"][metric] = {
            "cell_values_by_seed": {
                str(seed): {
                    method: float(cells[row_index, column_index])
                    for column_index, method in enumerate(CELL_NAMES)
                }
                for row_index, seed in enumerate(seeds)
            },
            "effects": {
                name: summarize_effect(
                    values,
                    repeats=bootstrap_repeats,
                    random_seed=bootstrap_seed + metric_index,
                )
                for name, values in effects.items()
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute descriptive paired 2x2 FMC-PC factorial effects."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenario", default="valid_signal_self")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--bootstrap-repeats", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    args = parser.parse_args()
    result = analyze_summary(
        read_json(args.summary),
        scenario=args.scenario,
        metrics=tuple(args.metrics),
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_json(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
