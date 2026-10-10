from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_decompose_expert_cv import context_frame, make_loader, predict, raw_batch  # noqa: E402
from run_ul_catboost_cv import load_folds  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.engine import smooth_probabilities  # noqa: E402
from rcbd_har.metrics import classification_metrics  # noqa: E402
from rcbd_har.models.decompose_expert import (  # noqa: E402
    TargetedSpectralCoordinationExpert,
)
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def select_competitor_classes(
    main_predictions: np.ndarray,
    targets: np.ndarray,
    focus_classes: tuple[int, ...],
    competitors_per_focus: int,
) -> list[int]:
    """Select hard negative labels using only the supplied training OOF rows."""

    main = np.asarray(main_predictions, dtype=np.int64)
    truth = np.asarray(targets, dtype=np.int64)
    if main.shape != truth.shape or main.ndim != 1:
        raise ValueError("Competitor selection inputs must align row-for-row")
    selected: set[int] = set()
    excluded = set(focus_classes)
    num_classes = int(max(main.max(initial=0), truth.max(initial=0))) + 1
    for focus in focus_classes:
        counts = np.zeros(num_classes, dtype=np.int64)
        predicted_for_focus = main[(truth == focus) & (main != truth)]
        true_for_focus_prediction = truth[(main == focus) & (main != truth)]
        counts += np.bincount(predicted_for_focus, minlength=num_classes)
        counts += np.bincount(true_for_focus_prediction, minlength=num_classes)
        order = sorted(
            (index for index in range(num_classes) if index not in excluded),
            key=lambda index: (-int(counts[index]), index),
        )
        selected.update(order[:competitors_per_focus])
    return sorted(selected)


def targeted_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    log_probabilities = F.log_softmax(logits, dim=1)
    rows = torch.arange(len(targets), device=targets.device)
    log_true = log_probabilities[rows, targets]
    true_probability = log_true.exp()
    return (
        class_weights[targets] * (1.0 - true_probability).pow(gamma) * -log_true
    ).mean()


def supervised_contrastive_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    relevant: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    selected = torch.nonzero(relevant, as_tuple=False).flatten()
    if len(selected) < 2:
        return features.sum() * 0.0
    values = F.normalize(features[selected], dim=1)
    labels = targets[selected]
    logits = values @ values.T / temperature
    identity = torch.eye(len(selected), dtype=torch.bool, device=features.device)
    positives = (labels[:, None] == labels[None, :]) & ~identity
    valid = positives.any(dim=1)
    if not valid.any():
        return features.sum() * 0.0
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(identity, 0.0)
    log_probabilities = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    mean_positive = (log_probabilities * positives).sum(dim=1) / positives.sum(
        dim=1
    ).clamp_min(1)
    return -mean_positive[valid].mean()


def train_model(
    frame,
    train_rows: np.ndarray,
    *,
    num_classes: int,
    focus_classes: tuple[int, ...],
    competitor_classes: list[int],
    args,
    device: torch.device,
    description: str,
) -> nn.Module:
    train_loader = make_loader(
        frame,
        train_rows,
        batch_size=args.batch_size,
        workers=args.workers,
        training=True,
    )
    model = TargetedSpectralCoordinationExpert(
        num_classes,
        embedding_dim=args.embedding_dim,
        recurrent_dim=args.recurrent_dim,
        spectral_dim=args.spectral_dim,
        coordination_dim=args.coordination_dim,
    ).to(device)
    counts = np.bincount(frame.iloc[train_rows].label, minlength=num_classes)
    class_weights = np.sqrt(len(train_rows) / (num_classes * np.maximum(counts, 1)))
    class_weights[list(focus_classes)] *= args.focus_multiplier
    class_weights[competitor_classes] *= args.competitor_multiplier
    class_weights_tensor = torch.as_tensor(
        class_weights, dtype=torch.float32, device=device
    )
    relevant_classes = torch.zeros(num_classes, dtype=torch.bool, device=device)
    relevant_classes[list(focus_classes) + competitor_classes] = True
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda")
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        examples = 0
        for batch in train_loader:
            values = raw_batch(batch).to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)  # type: ignore[union-attr]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                representation = model.encode(values)
                logits = model.classifier(representation)
                classification_loss = targeted_focal_loss(
                    logits, targets, class_weights_tensor, args.focal_gamma
                )
                contrastive_loss = supervised_contrastive_loss(
                    representation,
                    targets,
                    relevant_classes[targets],
                    args.contrastive_temperature,
                )
                loss = classification_loss + args.contrastive_weight * contrastive_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(targets)
            examples += len(targets)
        scheduler.step()
        print(
            f"{description} epoch {epoch + 1}/{args.epochs}: "
            f"loss={loss_sum / examples:.6f}",
            flush=True,
        )
    return model


def predict_rows(
    model: nn.Module,
    frame,
    rows: np.ndarray,
    args,
    device: torch.device,
) -> np.ndarray:
    loader: DataLoader = make_loader(
        frame,
        rows,
        batch_size=args.batch_size,
        workers=args.workers,
        training=False,
    )
    return predict(model, loader, device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict targeted spectral-coordination expert for WEAR"
    )
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--output", default="outputs/wear_targeted_spectral_seed0")
    parser.add_argument(
        "--inner-output-template",
        default="outputs/wear_targeted_spectral_inner_outer{fold}_seed0",
    )
    parser.add_argument(
        "--routing-output", default="outputs/wear_da_ccfre_seed0"
    )
    parser.add_argument(
        "--inner-summary-template",
        default="outputs/wear_null_residual_meta_seed0/fold_{fold}/summary.json",
    )
    parser.add_argument("--folds")
    parser.add_argument("--fold-indices", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--context-seconds", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--recurrent-dim", type=int, default=128)
    parser.add_argument("--spectral-dim", type=int, default=64)
    parser.add_argument("--coordination-dim", type=int, default=32)
    parser.add_argument("--focus-classes", nargs="+", type=int, default=[17, 18])
    parser.add_argument("--competitors-per-focus", type=int, default=2)
    parser.add_argument("--focus-multiplier", type=float, default=2.0)
    parser.add_argument("--competitor-multiplier", type=float, default=1.25)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--smooth-radius", type=int, default=10)
    parser.add_argument("--smooth-sigma", type=float, default=6.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("The targeted spectral expert requires a CUDA device")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    original_frame = load_manifest(args.manifest).reset_index(drop=True)
    frame = context_frame(original_frame, args.context_seconds, args.sample_rate)
    participant_values = frame.participant.astype(str).to_numpy()
    participants = sorted(np.unique(participant_values), key=participant_sort_key)
    folds = load_folds(args.folds, participant_values.tolist(), 3)
    selected_folds = (
        {value for value in args.fold_indices}
        if args.fold_indices
        else set(range(1, len(folds) + 1))
    )
    if not selected_folds or any(value < 1 or value > len(folds) for value in selected_folds):
        raise ValueError("Fold indices are outside the available folds")
    num_classes = int(frame.label.max()) + 1
    focus_classes = tuple(sorted(set(args.focus_classes)))
    if any(value < 0 or value >= num_classes for value in focus_classes):
        raise ValueError("Focus classes are outside the label space")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_json({"test_folds": folds}, output / "folds.json")
    outer_oof = np.full((len(frame), num_classes), np.nan, dtype=np.float32)
    fold_summaries: list[dict[str, object]] = []
    routing_root = Path(args.routing_output).resolve()

    for fold_index, test_people in enumerate(folds, start=1):
        if fold_index not in selected_folds:
            continue
        started = time.time()
        train_people = [value for value in participants if value not in test_people]
        train_rows = np.flatnonzero(np.isin(participant_values, train_people))
        test_rows = np.flatnonzero(np.isin(participant_values, test_people))
        routing = np.load(routing_root / f"fold_{fold_index}" / "inner_routing_oof.npz")
        inner_rows = routing["row_indices"].astype(np.int64)
        inner_targets = routing["targets"].astype(np.int64)
        inner_main = routing["main_predictions"].astype(np.int64)
        if not np.array_equal(inner_targets, frame.iloc[inner_rows].label.to_numpy(np.int64)):
            raise ValueError(f"Inner routing rows do not align in fold {fold_index}")
        if set(frame.iloc[inner_rows].participant.astype(str)) != set(train_people):
            raise ValueError(f"Inner routing participants do not align in fold {fold_index}")

        inner_summary = load_json(
            Path(args.inner_summary_template.format(fold=fold_index)).resolve()
        )
        inner_folds = inner_summary["inner_folds"]
        inner_oof = np.full((len(inner_rows), num_classes), np.nan, dtype=np.float32)
        inner_competitors: list[dict[str, object]] = []
        inner_people = frame.iloc[inner_rows].participant.astype(str).to_numpy()
        for inner_index, validation_people in enumerate(inner_folds, start=1):
            validation = np.isin(inner_people, validation_people)
            training = ~validation
            competitors = select_competitor_classes(
                inner_main[training],
                inner_targets[training],
                focus_classes,
                args.competitors_per_focus,
            )
            model = train_model(
                frame,
                inner_rows[training],
                num_classes=num_classes,
                focus_classes=focus_classes,
                competitor_classes=competitors,
                args=args,
                device=device,
                description=f"outer {fold_index} inner {inner_index}",
            )
            inner_oof[validation] = predict_rows(
                model, frame, inner_rows[validation], args, device
            )
            inner_competitors.append(
                {
                    "inner_fold": inner_index,
                    "validation_participants": validation_people,
                    "competitor_classes": competitors,
                }
            )
            del model
            torch.cuda.empty_cache()
        if not np.isfinite(inner_oof).all():
            raise RuntimeError(f"Targeted inner OOF is incomplete in fold {fold_index}")
        inner_frame_for_smoothing = original_frame.iloc[inner_rows].reset_index(drop=True)
        inner_smoothed = smooth_probabilities(
            inner_oof,
            inner_frame_for_smoothing,
            args.smooth_radius,
            args.smooth_sigma,
        )
        inner_output = Path(args.inner_output_template.format(fold=fold_index)).resolve()
        inner_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            inner_output / "oof_predictions.npz",
            row_indices=inner_rows,
            probabilities=inner_oof,
            smoothed_probabilities=inner_smoothed,
            targets=inner_targets,
        )

        outer_competitors = select_competitor_classes(
            inner_main,
            inner_targets,
            focus_classes,
            args.competitors_per_focus,
        )
        outer_model = train_model(
            frame,
            train_rows,
            num_classes=num_classes,
            focus_classes=focus_classes,
            competitor_classes=outer_competitors,
            args=args,
            device=device,
            description=f"outer {fold_index}",
        )
        outer_oof[test_rows] = predict_rows(outer_model, frame, test_rows, args, device)
        fold_dir = output / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(outer_model.state_dict(), fold_dir / "model.pt")
        test_frame = original_frame.iloc[test_rows].reset_index(drop=True)
        smoothed_test = smooth_probabilities(
            outer_oof[test_rows], test_frame, args.smooth_radius, args.smooth_sigma
        )
        fold_summary = {
            "fold": fold_index,
            "train_participants": train_people,
            "test_participants": test_people,
            "focus_classes": focus_classes,
            "outer_competitor_classes": outer_competitors,
            "inner_competitor_selection": inner_competitors,
            "inner_center_smoothed": classification_metrics(
                inner_targets, inner_smoothed.argmax(1), num_classes
            ),
            "test_center_smoothed": classification_metrics(
                test_frame.label.to_numpy(np.int64), smoothed_test.argmax(1), num_classes
            ),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        save_json(fold_summary, fold_dir / "summary.json")
        fold_summaries.append(fold_summary)
        print(
            f"fold {fold_index}: competitors={outer_competitors}; "
            f"inner={fold_summary['inner_center_smoothed']['macro_f1']:.4%}; "
            f"test={fold_summary['test_center_smoothed']['macro_f1']:.4%}",
            flush=True,
        )
        del outer_model
        torch.cuda.empty_cache()

    evaluated = np.isfinite(outer_oof).all(axis=1)
    evaluated_frame = original_frame.loc[evaluated].reset_index(drop=True)
    probabilities = outer_oof[evaluated]
    smoothed = smooth_probabilities(
        probabilities, evaluated_frame, args.smooth_radius, args.smooth_sigma
    )
    summary = {
        "method": "Targeted Spectral-Phase Coordination Expert",
        "strict_outer_test_label_free": True,
        "competitor_selection": "corresponding training-participant main-model OOF only",
        "seed": args.seed,
        "focus_classes": focus_classes,
        "context_seconds": args.context_seconds,
        "evaluated_folds": sorted(selected_folds),
        "folds": fold_summaries,
        "center_raw": classification_metrics(
            evaluated_frame.label.to_numpy(np.int64), probabilities.argmax(1), num_classes
        ),
        "center_smoothed": classification_metrics(
            evaluated_frame.label.to_numpy(np.int64), smoothed.argmax(1), num_classes
        ),
    }
    save_json(summary, output / "summary.json")
    np.savez_compressed(
        output / "oof_predictions.npz",
        row_indices=np.flatnonzero(evaluated),
        probabilities=probabilities,
        smoothed_probabilities=smoothed,
        targets=evaluated_frame.label.to_numpy(np.int64),
    )
    print(
        f"overall targeted expert center={summary['center_smoothed']['macro_f1']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
