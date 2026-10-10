from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_ccfre_cv import load_json  # noqa: E402
from run_decompose_expert_cv import context_frame, make_loader, predict  # noqa: E402
from run_null_aware_calibration_cv import fit_predict_split, load_participant  # noqa: E402
from run_null_residual_cv import predict_ul_pairs  # noqa: E402
from run_prior_alignment_cv import align_participant  # noqa: E402
from run_protocol_decoder_cv import minimum_count_vector  # noqa: E402
from run_ul_catboost_cv import load_folds  # noqa: E402
from rcbd_har.challenge_features import feature_names  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.engine import smooth_probabilities  # noqa: E402
from rcbd_har.models.decompose_expert import DecomposeCoordinationExpert  # noqa: E402
from rcbd_har.participant_calibration import (  # noqa: E402
    amplitude_feature_indices,
    apply_null_activity_gate,
    apply_participant_support_fallback,
)
from rcbd_har.prior_alignment import mean_participant_prior  # noqa: E402
from rcbd_har.protocol_decoder import protocol_constrained_decode  # noqa: E402
from rcbd_har.protocol_gate import (  # noqa: E402
    apply_protocol_gate,
    build_protocol_gate_features,
    hard_vote_with_probability_tiebreak,
)
from rcbd_har.residual_expert import (  # noqa: E402
    aggregate_pair_segment_scores,
    apply_residual_expert_gate,
    build_residual_expert_features,
    residual_route_scope,
)
from rcbd_har.temporal_decoder import estimate_transition_matrix  # noqa: E402
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


SEEDS = (0, 42, 3407, 114514)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_blind_manifest(frame, expected_people: list[str]) -> None:
    people = sorted(frame.participant.astype(str).unique(), key=participant_sort_key)
    if people != expected_people:
        raise ValueError(f"Unexpected external participants: {people}")
    labels = frame.label.to_numpy(dtype=np.int64)
    if not np.all(labels == -1):
        raise RuntimeError("Blind inference refuses a manifest containing revealed labels")
    if not np.array_equal(frame.index.to_numpy(), np.arange(len(frame))):
        raise ValueError("External manifest must have a contiguous default index")


def ordered_feature_data(feature_root: Path, people: list[str], row_count: int):
    data = {person: load_participant(feature_root, person) for person in people}
    rows = np.concatenate([data[person]["row_indices"] for person in people]).astype(
        np.int64
    )
    values = np.concatenate([data[person]["features"] for person in people])
    order = np.argsort(rows)
    if not np.array_equal(rows[order], np.arange(row_count)):
        raise ValueError("External feature rows do not align with the blind manifest")
    return data, values[order]


def infer_cstg_member(
    *,
    seed: int,
    fold_index: int,
    train_people: list[str],
    dev_frame,
    external_frame,
    external_features: np.ndarray,
    num_classes: int,
    args,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    from catboost import CatBoostClassifier

    base_root = Path(args.ul_template.format(seed=seed)).resolve()
    cstg_root = Path(args.cstg_template.format(seed=seed)).resolve()
    model = CatBoostClassifier()
    model.load_model(str(base_root / f"fold_{fold_index}" / "model.cbm"))
    pair_probabilities = predict_ul_pairs(model, external_features, num_classes)
    raw_probabilities = pair_probabilities.mean(axis=1).astype(np.float32)
    base_probabilities = smooth_probabilities(
        raw_probabilities, external_frame, args.smooth_radius, args.smooth_sigma
    )
    base_predictions = base_probabilities.argmax(axis=1).astype(np.int64)

    summary = load_json(cstg_root / f"fold_{fold_index}" / "summary.json")
    prior_configuration = summary["prior_configuration"]
    protocol_configuration = summary["protocol_configuration"]
    train_mask = dev_frame.participant.astype(str).isin(train_people).to_numpy()
    source_prior = mean_participant_prior(
        dev_frame.label.to_numpy(np.int64),
        dev_frame.participant.astype(str).to_numpy(),
        train_people,
        num_classes,
    )
    transition = estimate_transition_matrix(
        dev_frame.loc[train_mask, "label"].to_numpy(np.int64),
        dev_frame.loc[train_mask].reset_index(drop=True),
        num_classes,
        args.laplace,
    )
    participant_values = external_frame.participant.astype(str).to_numpy()
    aligned = np.empty_like(raw_probabilities)
    protocol = np.full(len(external_frame), -1, dtype=np.int64)
    alignment_details: dict[str, object] = {}
    for person in sorted(np.unique(participant_values), key=participant_sort_key):
        selected = participant_values == person
        person_frame = external_frame.loc[selected].reset_index(drop=True)
        person_aligned, estimated_null, target_prior = align_participant(
            raw_probabilities[selected],
            person_frame,
            source_prior,
            str(prior_configuration["null_estimator"]),
            float(prior_configuration["rho"]),
            float(prior_configuration["beta"]),
            args,
        )
        aligned[selected] = person_aligned
        decoded = protocol_constrained_decode(
            person_aligned,
            person_frame,
            transition,
            float(protocol_configuration["strength"]),
            minimum_count_vector(
                num_classes, int(protocol_configuration["minimum_count"])
            ),
        )
        protocol[selected] = decoded.predictions
        alignment_details[person] = {
            "estimated_null": float(estimated_null),
            "target_prior": target_prior.tolist(),
        }

    gate_features, _ = build_protocol_gate_features(
        raw_probabilities,
        aligned,
        pair_probabilities,
        base_predictions,
        protocol,
        external_frame,
    )
    gate = CatBoostClassifier()
    gate.load_model(str(cstg_root / f"fold_{fold_index}" / "gate.cbm"))
    base_better = gate.predict_proba(gate_features)[:, 1].astype(np.float32)
    gated = apply_protocol_gate(
        base_predictions,
        protocol,
        base_better,
        float(summary["selected_threshold"]),
    )
    diagnostics = {
        "seed": seed,
        "base_protocol_disagreement": float(np.mean(base_predictions != protocol)),
        "gate_switch_from_protocol": float(np.mean(gated != protocol)),
        "alignment": alignment_details,
    }
    return gated, aligned.astype(np.float32), diagnostics


def infer_decompose_experts(
    external_frame,
    num_classes: int,
    expert_root: Path,
    folds: list[list[str]],
    args,
) -> list[np.ndarray]:
    if not torch.cuda.is_available():
        raise RuntimeError("Blind raw-expert inference requires CUDA")
    device = torch.device("cuda")
    framed = context_frame(external_frame, args.context_seconds, args.sample_rate)
    loader = make_loader(
        framed,
        np.arange(len(framed)),
        batch_size=args.batch_size,
        workers=args.workers,
        training=False,
    )
    outputs: list[np.ndarray] = []
    for fold_index in range(1, len(folds) + 1):
        model = DecomposeCoordinationExpert(
            num_classes,
            embedding_dim=args.embedding_dim,
            recurrent_dim=args.recurrent_dim,
        ).to(device)
        state = torch.load(
            expert_root / f"fold_{fold_index}" / "model.pt",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state)
        raw = predict(model, loader, device)
        outputs.append(
            smooth_probabilities(
                raw, external_frame, args.smooth_radius, args.smooth_sigma
            ).astype(np.float32)
        )
        del model
        torch.cuda.empty_cache()
    return outputs


def infer_ccfre_fold(
    *,
    fold_index: int,
    member_predictions: np.ndarray,
    member_support: np.ndarray,
    expert_probabilities: np.ndarray,
    external_frame,
    ccfre_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    from catboost import CatBoostClassifier

    main_support = member_support.mean(axis=0).astype(np.float32)
    main = hard_vote_with_probability_tiebreak(member_predictions, member_support)
    expert = expert_probabilities.argmax(axis=1).astype(np.int64)
    features, names = build_residual_expert_features(
        main_support,
        expert_probabilities,
        main,
        expert,
        external_frame,
        member_probabilities=member_support,
        member_predictions=member_predictions,
    )
    summary = load_json(ccfre_root / f"fold_{fold_index}" / "summary.json")
    if int(summary["feature_count"]) != len(names):
        raise RuntimeError(f"CCFRE feature count changed in fold {fold_index}")
    gate = CatBoostClassifier()
    gate.load_model(str(ccfre_root / f"fold_{fold_index}" / "gate.cbm"))
    main_better = gate.predict_proba(features)[:, 1].astype(np.float32)
    eligible = residual_route_scope(
        main,
        expert,
        scope=str(summary["selected_routing_scope"]),
        focus_classes=[17, 18],
    )
    main_better[~eligible] = 1.0
    if float(summary["selected_pair_strength"]) != 0.0:
        raise RuntimeError("Frozen SA-DA-CCFRE unexpectedly requires pair-risk fitting")
    main_better = aggregate_pair_segment_scores(
        main,
        expert,
        main_better,
        external_frame,
        blend=float(summary["selected_segment_blend"]),
        minimum_length=int(summary["selected_segment_minimum_length"]),
    )
    gated = apply_residual_expert_gate(
        main,
        expert,
        main_better,
        float(summary["selected_threshold"]),
    )
    diagnostics = {
        "fold": fold_index,
        "main_expert_disagreement": float(np.mean(main != expert)),
        "expert_switch_fraction": float(np.mean(gated != main)),
        "selected_threshold": float(summary["selected_threshold"]),
        "selected_segment_blend": float(summary["selected_segment_blend"]),
        "selected_segment_minimum_length": int(
            summary["selected_segment_minimum_length"]
        ),
    }
    return gated, main_support, diagnostics


def infer_nag_fold(
    *,
    fold_index: int,
    train_people: list[str],
    external_people: list[str],
    all_feature_data: dict[str, dict[str, np.ndarray]],
    amplitude_indices: np.ndarray,
    main_predictions: np.ndarray,
    external_frame,
    nag_root: Path,
    args,
) -> tuple[np.ndarray, dict[str, object]]:
    summary = load_json(nag_root / f"fold_{fold_index}" / "summary.json")
    selected = summary["selected_configuration"]
    if not bool(selected["enabled"]):
        return main_predictions.copy(), {
            "fold": fold_index,
            "enabled": False,
            "switch_fraction": 0.0,
            "unsupported_participants": [],
        }
    rows, probabilities, shifts = fit_predict_split(
        train_people,
        external_people,
        all_feature_data,
        amplitude_indices,
        float(selected["strength"]),
        args,
    )
    order = np.argsort(rows)
    if not np.array_equal(rows[order], np.arange(len(external_frame))):
        raise ValueError(f"NAG rows do not align in fold {fold_index}")
    probabilities = probabilities[order]
    shifts = shifts[order]
    unsafe = apply_null_activity_gate(
        main_predictions,
        probabilities,
        shifts,
        float(selected["promote_margin"]),
        float(selected["demote_margin"]),
        float(selected["minimum_shift"]),
    )
    calibrated, unsafe_rates, unsupported = apply_participant_support_fallback(
        main_predictions,
        unsafe,
        external_frame.participant.astype(str).to_numpy(),
        float(selected["inner_participant_switch_ceiling"]),
    )
    return calibrated, {
        "fold": fold_index,
        "enabled": True,
        "switch_fraction": float(np.mean(calibrated != main_predictions)),
        "unsafe_switch_rates": {key: float(value) for key, value in unsafe_rates.items()},
        "unsupported_participants": unsupported,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label-blind external WEAR 2024 inference for frozen SA-DA-CCFRE/NAG"
    )
    parser.add_argument("--dev-manifest", default="data/wear/manifest.csv")
    parser.add_argument("--dev-features", default="data/wear/challenge_features")
    parser.add_argument(
        "--external-manifest", default="data/wear/external_2024/unlabeled/manifest.csv"
    )
    parser.add_argument(
        "--external-features",
        default="data/wear/external_2024/unlabeled/challenge_features",
    )
    parser.add_argument("--output", default="outputs/wear_external_2024")
    parser.add_argument("--folds", default="outputs/wear_sa_da_ccfre_focus_seed0/folds.json")
    parser.add_argument("--ul-template", default="outputs/wear_ul_catboost_seed{seed}")
    parser.add_argument("--cstg-template", default="outputs/wear_cstg_safe_seed{seed}")
    parser.add_argument("--expert-root", default="outputs/wear_decompose_expert_4s_seed0")
    parser.add_argument("--ccfre-root", default="outputs/wear_sa_da_ccfre_focus_seed0")
    parser.add_argument("--nag-root", default="outputs/wear_nag_psf15_seed0")
    parser.add_argument("--smooth-radius", type=int, default=10)
    parser.add_argument("--smooth-sigma", type=float, default=6.0)
    parser.add_argument("--laplace", type=float, default=1e-2)
    parser.add_argument("--context-seconds", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--recurrent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--border-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-factor", type=float, default=0.5)
    parser.add_argument("--maximum-factor", type=float, default=2.0)
    parser.add_argument("--task-type", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--thread-count", type=int, default=-1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dev_frame = load_manifest(args.dev_manifest).reset_index(drop=True)
    external_frame = load_manifest(args.external_manifest).reset_index(drop=True)
    external_people = ["sbj_18", "sbj_19", "sbj_20", "sbj_21"]
    validate_blind_manifest(external_frame, external_people)
    num_classes = int(dev_frame.label.max()) + 1
    args.num_classes = num_classes
    dev_people = sorted(
        dev_frame.participant.astype(str).unique(), key=participant_sort_key
    )
    folds = load_folds(args.folds, dev_frame.participant.astype(str).tolist(), 3)
    dev_feature_root = Path(args.dev_features).resolve()
    external_feature_root = Path(args.external_features).resolve()
    dev_data = {person: load_participant(dev_feature_root, person) for person in dev_people}
    external_data, external_features = ordered_feature_data(
        external_feature_root, external_people, len(external_frame)
    )

    all_feature_data = {**dev_data, **external_data}
    expert_probabilities = infer_decompose_experts(
        external_frame,
        num_classes,
        Path(args.expert_root).resolve(),
        folds,
        args,
    )
    fold_main: list[np.ndarray] = []
    fold_nag: list[np.ndarray] = []
    fold_support: list[np.ndarray] = []
    diagnostics: list[dict[str, object]] = []
    amplitude_indices = amplitude_feature_indices(feature_names())

    for fold_index, test_people in enumerate(folds, start=1):
        train_people = [person for person in dev_people if person not in test_people]
        members: list[np.ndarray] = []
        supports: list[np.ndarray] = []
        member_diagnostics: list[dict[str, object]] = []
        for seed in SEEDS:
            prediction, support, detail = infer_cstg_member(
                seed=seed,
                fold_index=fold_index,
                train_people=train_people,
                dev_frame=dev_frame,
                external_frame=external_frame,
                external_features=external_features,
                num_classes=num_classes,
                args=args,
            )
            members.append(prediction)
            supports.append(support)
            member_diagnostics.append(detail)
        main, support, ccfre_detail = infer_ccfre_fold(
            fold_index=fold_index,
            member_predictions=np.stack(members),
            member_support=np.stack(supports),
            expert_probabilities=expert_probabilities[fold_index - 1],
            external_frame=external_frame,
            ccfre_root=Path(args.ccfre_root).resolve(),
        )
        nag, nag_detail = infer_nag_fold(
            fold_index=fold_index,
            train_people=train_people,
            external_people=external_people,
            all_feature_data=all_feature_data,
            amplitude_indices=amplitude_indices,
            main_predictions=main,
            external_frame=external_frame,
            nag_root=Path(args.nag_root).resolve(),
            args=args,
        )
        fold_main.append(main)
        fold_nag.append(nag)
        fold_support.append(support)
        diagnostics.append(
            {
                "fold": fold_index,
                "development_train_participants": train_people,
                "cstg_members": member_diagnostics,
                "ccfre": ccfre_detail,
                "nag": nag_detail,
            }
        )
        print(
            f"fold {fold_index}: CCFRE expert switch="
            f"{ccfre_detail['expert_switch_fraction']:.2%}; "
            f"NAG switch={nag_detail['switch_fraction']:.2%}",
            flush=True,
        )

    support_stack = np.stack(fold_support)
    main_stack = np.stack(fold_main)
    nag_stack = np.stack(fold_nag)
    final_main = hard_vote_with_probability_tiebreak(main_stack, support_stack)
    final_nag = hard_vote_with_probability_tiebreak(nag_stack, support_stack)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions_blind.npz"
    np.savez_compressed(
        prediction_path,
        row_indices=np.arange(len(external_frame), dtype=np.int64),
        centers=external_frame.center.to_numpy(np.int64),
        participants=external_frame.participant.astype(str).to_numpy(),
        fold_main_predictions=main_stack,
        fold_nag_predictions=nag_stack,
        fold_probability_support=support_stack,
        main_predictions=final_main,
        nag_predictions=final_nag,
    )
    prediction_hash = sha256_file(prediction_path)
    summary = {
        "method": "frozen external WEAR 2024 label-blind inference",
        "external_participants": external_people,
        "external_window_count": len(external_frame),
        "external_labels_observed": False,
        "development_participants": dev_people,
        "seeds": list(SEEDS),
        "folds": diagnostics,
        "final_main_nag_disagreement": float(np.mean(final_main != final_nag)),
        "prediction_file": str(prediction_path),
        "prediction_sha256": prediction_hash,
    }
    save_json(summary, output / "blind_summary.json")
    (output / "predictions_blind.sha256").write_text(
        f"{prediction_hash}  {prediction_path.name}\n", encoding="ascii"
    )
    print(f"blind predictions saved: sha256={prediction_hash}", flush=True)


if __name__ == "__main__":
    main()
