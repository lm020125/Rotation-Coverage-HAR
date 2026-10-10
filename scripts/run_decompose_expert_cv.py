from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_ul_catboost_cv import load_folds  # noqa: E402
from rcbd_har.data import WindowManifestDataset, load_manifest  # noqa: E402
from rcbd_har.engine import smooth_probabilities  # noqa: E402
from rcbd_har.metrics import classification_metrics  # noqa: E402
from rcbd_har.models.decompose_expert import (  # noqa: E402
    DecomposeCoordinationExpert,
)
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


LIMBS = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")


def raw_batch(batch: dict[str, torch.Tensor | list[str]]) -> torch.Tensor:
    return torch.stack([batch[name] for name in LIMBS], dim=1)  # type: ignore[list-item]


def context_frame(frame, context_seconds: float, sample_rate: float):
    output = frame.copy()
    width = int(round(context_seconds * sample_rate))
    if width <= 1:
        raise ValueError("The raw expert context must contain at least two samples")
    output["start"] = output.center.astype(np.int64) - width // 2
    output["stop"] = output["start"] + width
    return output


def make_loader(
    frame,
    indices: np.ndarray,
    *,
    batch_size: int,
    workers: int,
    training: bool,
) -> DataLoader:
    dataset = WindowManifestDataset(
        frame,
        indices=indices.tolist(),
        normalization="scale_only",
        normalization_scale=8.0,
        missing_strategy="zero",
        limb_mode="four",
        swap_probability=0.5 if training else 0.0,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=training,
    )


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    for batch in loader:
        values = raw_batch(batch).to(device, non_blocking=True)
        probabilities = model(values).softmax(dim=1)
        parts.append(probabilities.cpu().numpy().astype(np.float32))
    return np.concatenate(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="WEAR DecomposeWHAR raw coordination expert")
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--output", default="outputs/wear_decompose_expert_seed0")
    parser.add_argument("--folds")
    parser.add_argument("--fold-indices", nargs="+", type=int)
    parser.add_argument(
        "--participant-subset",
        nargs="+",
        help="Optional participant subset for nested inner-OOF training",
    )
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
    parser.add_argument("--smooth-radius", type=int, default=10)
    parser.add_argument("--smooth-sigma", type=float, default=6.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("The raw coordination expert requires a CUDA device")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    full_frame = load_manifest(args.manifest).reset_index(drop=True)
    if args.participant_subset:
        requested = {str(value) for value in args.participant_subset}
        available = set(full_frame.participant.astype(str))
        if missing := requested.difference(available):
            raise ValueError(f"Unknown participant subset: {sorted(missing)}")
        selected_rows = np.flatnonzero(
            full_frame.participant.astype(str).isin(requested).to_numpy()
        )
        original_frame = full_frame.iloc[selected_rows].reset_index(drop=True)
        global_rows = selected_rows.astype(np.int64)
    else:
        original_frame = full_frame
        global_rows = np.arange(len(full_frame), dtype=np.int64)
    frame = context_frame(original_frame, args.context_seconds, args.sample_rate)
    participant_values = frame.participant.astype(str).to_numpy()
    participants = sorted(np.unique(participant_values), key=participant_sort_key)
    folds = load_folds(args.folds, participant_values.tolist(), 3)
    selected = (
        [value - 1 for value in args.fold_indices]
        if args.fold_indices
        else list(range(len(folds)))
    )
    if not selected or any(value < 0 or value >= len(folds) for value in selected):
        raise ValueError("Fold indices are outside the available three folds")
    num_classes = int(frame.label.max()) + 1
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_json({"test_folds": folds}, output / "folds.json")
    oof = np.full((len(frame), num_classes), np.nan, dtype=np.float32)
    fold_summaries: list[dict[str, object]] = []

    for fold_index in selected:
        started = time.time()
        test_people = folds[fold_index]
        train_people = [value for value in participants if value not in test_people]
        train_rows = np.flatnonzero(np.isin(participant_values, train_people))
        test_rows = np.flatnonzero(np.isin(participant_values, test_people))
        train_loader = make_loader(
            frame,
            train_rows,
            batch_size=args.batch_size,
            workers=args.workers,
            training=True,
        )
        test_loader = make_loader(
            frame,
            test_rows,
            batch_size=args.batch_size,
            workers=args.workers,
            training=False,
        )
        model = DecomposeCoordinationExpert(
            num_classes,
            embedding_dim=args.embedding_dim,
            recurrent_dim=args.recurrent_dim,
        ).to(device)
        counts = np.bincount(frame.iloc[train_rows].label, minlength=num_classes)
        class_weights = np.sqrt(len(train_rows) / (num_classes * np.maximum(counts, 1)))
        criterion = nn.CrossEntropyLoss(
            weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device),
            label_smoothing=0.05,
        )
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
                    loss = criterion(model(values), targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * len(targets)
                examples += len(targets)
            scheduler.step()
            print(
                f"fold {fold_index + 1} epoch {epoch + 1}/{args.epochs}: "
                f"loss={loss_sum / examples:.6f}",
                flush=True,
            )

        probabilities = predict(model, test_loader, device)
        if len(probabilities) != len(test_rows):
            raise RuntimeError("Raw expert prediction rows are incomplete")
        oof[test_rows] = probabilities
        test_frame = original_frame.iloc[test_rows].reset_index(drop=True)
        smoothed = smooth_probabilities(
            probabilities, test_frame, args.smooth_radius, args.smooth_sigma
        )
        raw_metrics = classification_metrics(
            test_frame.label.to_numpy(np.int64), probabilities.argmax(1), num_classes
        )
        smooth_metrics = classification_metrics(
            test_frame.label.to_numpy(np.int64), smoothed.argmax(1), num_classes
        )
        fold_dir = output / f"fold_{fold_index + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), fold_dir / "model.pt")
        fold_summary = {
            "fold": fold_index + 1,
            "train_participants": train_people,
            "test_participants": test_people,
            "raw_center": raw_metrics,
            "smoothed_center": smooth_metrics,
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        save_json(fold_summary, fold_dir / "summary.json")
        fold_summaries.append(fold_summary)
        print(
            f"fold {fold_index + 1}: raw={raw_metrics['macro_f1']:.4%}; "
            f"smoothed={smooth_metrics['macro_f1']:.4%}",
            flush=True,
        )
        del model, optimizer, train_loader, test_loader
        torch.cuda.empty_cache()

    evaluated = np.isfinite(oof).all(axis=1)
    evaluated_frame = original_frame.loc[evaluated].reset_index(drop=True)
    probabilities = oof[evaluated]
    smoothed = smooth_probabilities(
        probabilities, evaluated_frame, args.smooth_radius, args.smooth_sigma
    )
    summary = {
        "method": "DecomposeWHAR-inspired full-body raw coordination expert",
        "source": "https://github.com/Anakin2555/DecomposeWHAR",
        "seed": args.seed,
        "context_seconds": args.context_seconds,
        "participant_subset": args.participant_subset,
        "evaluated_folds": [value + 1 for value in selected],
        "folds": fold_summaries,
        "center_raw": classification_metrics(
            evaluated_frame.label.to_numpy(np.int64),
            probabilities.argmax(1),
            num_classes,
        ),
        "center_smoothed": classification_metrics(
            evaluated_frame.label.to_numpy(np.int64), smoothed.argmax(1), num_classes
        ),
    }
    save_json(summary, output / "summary.json")
    np.savez_compressed(
        output / "oof_predictions.npz",
        row_indices=global_rows[np.flatnonzero(evaluated)],
        probabilities=probabilities,
        smoothed_probabilities=smoothed,
        targets=evaluated_frame.label.to_numpy(np.int64),
    )
    print(
        f"overall smoothed center={summary['center_smoothed']['macro_f1']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
