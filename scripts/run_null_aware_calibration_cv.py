from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_temporal_decoder_cv import evaluate_labels  # noqa: E402
from run_ul_catboost_cv import load_folds, make_grouped_folds  # noqa: E402
from rcbd_har.challenge_features import feature_names  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.metrics import classification_metrics  # noqa: E402
from rcbd_har.participant_calibration import (  # noqa: E402
    amplitude_feature_indices,
    apply_null_activity_gate,
    apply_participant_support_fallback,
    calibrate_amplitude_features,
    compose_null_activity_probabilities,
    fit_amplitude_reference,
    select_null_activity_rule,
)
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


def load_participant(feature_dir: Path, participant: str) -> dict[str, np.ndarray]:
    with np.load(feature_dir / f"{participant}.npz", allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def aligned_probabilities(model, values: np.ndarray, classes: list[int]) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(values), dtype=np.float32)
    output = np.zeros((len(values), len(classes)), dtype=np.float32)
    positions = {value: index for index, value in enumerate(classes)}
    for source, label in enumerate(np.asarray(model.classes_, dtype=int)):
        output[:, positions[int(label)]] = probabilities[:, source]
    return output


def train_two_stage(features: np.ndarray, labels: np.ndarray, args):
    from catboost import CatBoostClassifier

    common = dict(
        iterations=args.iterations,
        depth=args.depth,
        border_count=args.border_count,
        auto_class_weights="Balanced",
        random_seed=args.seed,
        task_type=args.task_type,
        devices=args.devices if args.task_type == "GPU" else None,
        thread_count=args.thread_count,
        verbose=False,
        allow_writing_files=False,
    )
    activity = CatBoostClassifier(loss_function="Logloss", **common)
    activity.fit(features, (labels != 0).astype(np.int64))
    active = labels != 0
    conditional = CatBoostClassifier(loss_function="MultiClass", **common)
    conditional.fit(features[active], labels[active])
    return activity, conditional


def fit_predict_split(
    training_people: list[str],
    validation_people: list[str],
    data: dict[str, dict[str, np.ndarray]],
    amplitude_indices: np.ndarray,
    strength: float,
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = fit_amplitude_reference(
        {person: data[person]["features"] for person in training_people},
        amplitude_indices,
    )
    train_features: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    for person in training_people:
        calibrated, _, _ = calibrate_amplitude_features(
            data[person]["features"],
            amplitude_indices,
            reference,
            strength,
            args.minimum_factor,
            args.maximum_factor,
        )
        train_features.append(calibrated)
        train_labels.append(data[person]["labels"].astype(np.int64))
    model_activity, model_conditional = train_two_stage(
        np.concatenate(train_features), np.concatenate(train_labels), args
    )
    output_rows: list[np.ndarray] = []
    output_probabilities: list[np.ndarray] = []
    output_shifts: list[np.ndarray] = []
    for person in validation_people:
        calibrated, shift, _ = calibrate_amplitude_features(
            data[person]["features"],
            amplitude_indices,
            reference,
            strength,
            args.minimum_factor,
            args.maximum_factor,
        )
        binary = aligned_probabilities(model_activity, calibrated, [0, 1])
        conditional = aligned_probabilities(
            model_conditional, calibrated, list(range(1, args.num_classes))
        )
        output_rows.append(data[person]["row_indices"].astype(np.int64))
        output_probabilities.append(compose_null_activity_probabilities(binary[:, 1], conditional))
        output_shifts.append(np.full(len(calibrated), shift, dtype=np.float32))
    return (
        np.concatenate(output_rows),
        np.concatenate(output_probabilities),
        np.concatenate(output_shifts),
    )


def cross_fitted_candidate(
    people: list[str],
    frame,
    data: dict[str, dict[str, np.ndarray]],
    amplitude_indices: np.ndarray,
    strength: float,
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = frame[frame.participant.astype(str).isin(people)]
    folds = make_grouped_folds(subset.participant.astype(str).tolist(), n_splits=3)
    rows = subset.index.to_numpy(dtype=np.int64)
    position = {int(row): index for index, row in enumerate(rows)}
    probabilities = np.full((len(rows), args.num_classes), np.nan, dtype=np.float32)
    shifts = np.full(len(rows), np.nan, dtype=np.float32)
    for validation_people in folds:
        training_people = [value for value in people if value not in validation_people]
        split_rows, split_probabilities, split_shifts = fit_predict_split(
            training_people,
            validation_people,
            data,
            amplitude_indices,
            strength,
            args,
        )
        destinations = np.asarray([position[int(row)] for row in split_rows], dtype=np.int64)
        probabilities[destinations] = split_probabilities
        shifts[destinations] = split_shifts
    if not np.isfinite(probabilities).all() or not np.isfinite(shifts).all():
        raise RuntimeError("Inner candidate predictions are incomplete")
    return rows, probabilities, shifts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict null-aware participant amplitude calibration"
    )
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--features", default="data/wear/challenge_features")
    parser.add_argument("--main-output", default="outputs/wear_sa_da_ccfre_focus_seed0")
    parser.add_argument("--output", default="outputs/wear_napc_seed0")
    parser.add_argument("--folds")
    parser.add_argument("--fold-indices", nargs="+", type=int)
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument(
        "--promote-margins",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 2.0],
    )
    parser.add_argument(
        "--demote-margins",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 2.0],
    )
    parser.add_argument(
        "--minimum-shifts", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.3]
    )
    parser.add_argument("--minimum-factor", type=float, default=0.5)
    parser.add_argument("--maximum-factor", type=float, default=2.0)
    parser.add_argument("--minimum-inner-gain", type=float, default=2e-4)
    parser.add_argument("--participant-switch-support-multiplier", type=float, default=1.5)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--border-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-type", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--thread-count", type=int, default=-1)
    args = parser.parse_args()
    if args.participant_switch_support_multiplier < 1.0:
        raise ValueError("Participant switch support multiplier must be at least 1")

    frame = load_manifest(args.manifest).reset_index(drop=True)
    args.num_classes = int(frame.label.max()) + 1
    targets = frame.label.to_numpy(dtype=np.int64)
    feature_dir = Path(args.features).resolve()
    main_root = Path(args.main_output).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    people = sorted(frame.participant.astype(str).unique().tolist(), key=participant_sort_key)
    folds = load_folds(
        args.folds or str(main_root / "folds.json"),
        frame.participant.astype(str).tolist(),
        3,
    )
    selected_folds = args.fold_indices if args.fold_indices else list(range(1, len(folds) + 1))
    save_json({"test_folds": folds}, output / "folds.json")
    names = feature_names()
    amplitude_indices = amplitude_feature_indices(names)
    if len(amplitude_indices) != 4 * 3 * 2 * 6 * 3:
        raise RuntimeError(f"Unexpected amplitude feature count: {len(amplitude_indices)}")
    data = {person: load_participant(feature_dir, person) for person in people}
    main_archive = np.load(main_root / "oof_predictions.npz")
    if not np.array_equal(main_archive["row_indices"], np.arange(len(frame))):
        raise ValueError("Main OOF rows do not align with the manifest")
    if not np.array_equal(main_archive["targets"], targets):
        raise ValueError("Main OOF labels do not align with the manifest")
    main_predictions = main_archive["gated_predictions"].astype(np.int64)
    calibrated_oof = np.full(len(frame), -1, dtype=np.int64)
    candidate_oof = np.full(len(frame), -1, dtype=np.int64)
    fold_summaries: list[dict[str, object]] = []

    for fold_index, test_people in enumerate(folds, start=1):
        if fold_index not in selected_folds:
            continue
        started = time.time()
        train_people = [value for value in people if value not in test_people]
        inner_archive = np.load(main_root / f"fold_{fold_index}" / "inner_routing_oof.npz")
        inner_rows = inner_archive["row_indices"].astype(np.int64)
        inner_targets = inner_archive["targets"].astype(np.int64)
        inner_main = inner_archive["gated_predictions"].astype(np.int64)
        if not np.array_equal(inner_targets, targets[inner_rows]):
            raise ValueError(f"Fold {fold_index} inner labels do not align")
        candidate_probabilities: dict[float, np.ndarray] = {}
        candidate_shifts: dict[float, np.ndarray] = {}
        for strength in args.strengths:
            rows, probabilities, shifts = cross_fitted_candidate(
                train_people, frame, data, amplitude_indices, strength, args
            )
            locations = {int(row): index for index, row in enumerate(rows)}
            order = np.asarray([locations[int(row)] for row in inner_rows], dtype=np.int64)
            candidate_probabilities[float(strength)] = probabilities[order]
            candidate_shifts[float(strength)] = shifts[order]
        selected, selection_scores = select_null_activity_rule(
            inner_main,
            candidate_probabilities,
            candidate_shifts,
            inner_targets,
            args.promote_margins,
            args.demote_margins,
            args.minimum_shifts,
            args.num_classes,
            args.minimum_inner_gain,
        )
        strength = float(selected["strength"])
        if bool(selected["enabled"]):
            inner_calibrated = apply_null_activity_gate(
                inner_main,
                candidate_probabilities[strength],
                candidate_shifts[strength],
                float(selected["promote_margin"]),
                float(selected["demote_margin"]),
                float(selected["minimum_shift"]),
            )
            inner_participants = frame.iloc[inner_rows].participant.astype(str).to_numpy()
            inner_switch_rates = {
                person: float(
                    np.mean(
                        inner_calibrated[inner_participants == person]
                        != inner_main[inner_participants == person]
                    )
                )
                for person in np.unique(inner_participants)
            }
            raw_switch_ceiling = max(inner_switch_rates.values())
            switch_ceiling = min(
                1.0, raw_switch_ceiling * args.participant_switch_support_multiplier
            )
        else:
            inner_switch_rates = {}
            raw_switch_ceiling = 0.0
            switch_ceiling = 0.0
        selected["inner_participant_raw_switch_ceiling"] = float(raw_switch_ceiling)
        selected["inner_participant_switch_ceiling"] = float(switch_ceiling)
        test_rows, test_probabilities, test_shifts = fit_predict_split(
            train_people,
            test_people,
            data,
            amplitude_indices,
            strength,
            args,
        )
        test_main = main_predictions[test_rows]
        test_candidate = test_probabilities.argmax(axis=1)
        if bool(selected["enabled"]):
            unsafe_calibrated = apply_null_activity_gate(
                test_main,
                test_probabilities,
                test_shifts,
                float(selected["promote_margin"]),
                float(selected["demote_margin"]),
                float(selected["minimum_shift"]),
            )
            test_participants = frame.iloc[test_rows].participant.astype(str).to_numpy()
            test_calibrated, unsafe_switch_rates, unsupported_participants = (
                apply_participant_support_fallback(
                    test_main,
                    unsafe_calibrated,
                    test_participants,
                    switch_ceiling,
                )
            )
        else:
            unsafe_calibrated = test_main.copy()
            test_calibrated = test_main.copy()
            unsafe_switch_rates = {
                person: 0.0 for person in frame.iloc[test_rows].participant.astype(str).unique()
            }
            unsupported_participants = []
        calibrated_oof[test_rows] = test_calibrated
        candidate_oof[test_rows] = test_candidate
        test_frame = frame.iloc[test_rows].reset_index(drop=True)
        test_result = evaluate_labels(test_calibrated, test_frame, args.num_classes)
        participant_results = {}
        for person in test_people:
            mask = test_frame.participant.astype(str).to_numpy() == person
            participant_results[person] = {
                "main_center_macro_f1": classification_metrics(
                    test_frame.loc[mask, "label"], test_main[mask], args.num_classes
                )["macro_f1"],
                "calibrated_center_macro_f1": classification_metrics(
                    test_frame.loc[mask, "label"], test_calibrated[mask], args.num_classes
                )["macro_f1"],
                "candidate_center_macro_f1": classification_metrics(
                    test_frame.loc[mask, "label"], test_candidate[mask], args.num_classes
                )["macro_f1"],
                "shift_score": float(test_shifts[mask][0]),
                "switch_fraction": float(np.mean(test_calibrated[mask] != test_main[mask])),
                "unsafe_switch_fraction": float(unsafe_switch_rates[person]),
                "support_fallback": person in unsupported_participants,
            }
        fold_summary = {
            "fold": fold_index,
            "train_participants": train_people,
            "test_participants": test_people,
            "selected_configuration": selected,
            "inner_participant_switch_rates": inner_switch_rates,
            "unsupported_test_participants": unsupported_participants,
            "selection_scores": selection_scores,
            "inner_candidate_center_macro_f1": classification_metrics(
                inner_targets,
                candidate_probabilities[strength].argmax(axis=1),
                args.num_classes,
            )["macro_f1"],
            "outer_switch_fraction": float(np.mean(test_calibrated != test_main)),
            "participants": participant_results,
            "test": test_result,
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        fold_dir = output / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_json(fold_summary, fold_dir / "summary.json")
        np.savez_compressed(
            fold_dir / "inner_oof.npz",
            row_indices=inner_rows,
            main_predictions=inner_main,
            candidate_probabilities=candidate_probabilities[strength],
            candidate_shift_scores=candidate_shifts[strength],
            targets=inner_targets,
        )
        fold_summaries.append(fold_summary)
        print(
            f"fold {fold_index}: enabled={selected['enabled']}; strength={strength:g}; "
            f"switch={fold_summary['outer_switch_fraction']:.2%}; "
            f"sample={test_result['samplewise']['macro_f1']:.4%}",
            flush=True,
        )

    evaluated = calibrated_oof >= 0
    evaluated_frame = frame.loc[evaluated].reset_index(drop=True)
    overall = evaluate_labels(calibrated_oof[evaluated], evaluated_frame, args.num_classes)
    summary = {
        "method": "Null-Activity Gate with Participant Support Fallback (NAG-PSF)",
        "strict_outer_test_label_free": True,
        "calibration_statistics": "per-participant unlabeled medians and outer-training equal-participant reference",
        "calibrated_feature_kinds": ["peak_to_peak", "iqr", "std"],
        "rule_selection": "outer-training cross-fitted inner OOF only with identity and participant-support fallback",
        "main_output": str(main_root),
        "minimum_inner_gain": args.minimum_inner_gain,
        "participant_switch_support_multiplier": args.participant_switch_support_multiplier,
        "evaluated_folds": selected_folds,
        "folds": fold_summaries,
        "overall": overall,
    }
    save_json(summary, output / "summary.json")
    np.savez_compressed(
        output / "oof_predictions.npz",
        row_indices=np.flatnonzero(evaluated),
        main_predictions=main_predictions[evaluated],
        candidate_predictions=candidate_oof[evaluated],
        calibrated_predictions=calibrated_oof[evaluated],
        targets=targets[evaluated],
    )
    print(
        f"overall center={overall['center']['macro_f1']:.4%}; "
        f"sample={overall['samplewise']['macro_f1']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
