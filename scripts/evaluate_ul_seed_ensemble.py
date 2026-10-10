from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_ul_catboost_cv import samplewise_targets  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.engine import expand_probabilities_to_samples, smooth_probabilities  # noqa: E402
from rcbd_har.metrics import classification_metrics  # noqa: E402
from rcbd_har.utils import save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an equal-weight UL seed ensemble")
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smooth-radius", type=int, default=10)
    parser.add_argument("--smooth-sigma", type=float, default=6.0)
    args = parser.parse_args()

    frame = load_manifest(args.manifest).reset_index(drop=True)
    targets = frame.label.to_numpy(np.int64)
    probabilities = []
    source_paths = []
    for raw_path in args.inputs:
        path = Path(raw_path).resolve()
        archive = np.load(path / "oof_predictions.npz")
        if not np.array_equal(archive["row_indices"], np.arange(len(frame))):
            raise ValueError(f"OOF rows do not align: {path}")
        if not np.array_equal(archive["targets"], targets):
            raise ValueError(f"OOF targets do not align: {path}")
        probabilities.append(archive["probabilities"].astype(np.float64))
        source_paths.append(str(path))
    ensemble = np.mean(probabilities, axis=0).astype(np.float32)
    num_classes = ensemble.shape[1]
    raw_metrics = classification_metrics(targets, ensemble.argmax(axis=1), num_classes)
    smoothed = smooth_probabilities(
        ensemble, frame, args.smooth_radius, args.smooth_sigma
    )
    smoothed_metrics = classification_metrics(
        targets, smoothed.argmax(axis=1), num_classes
    )
    expanded = expand_probabilities_to_samples(smoothed, frame)
    sample_targets, sample_predictions = samplewise_targets(frame, expanded)
    sample_metrics = classification_metrics(sample_targets, sample_predictions, num_classes)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "equal-weight UL CatBoost seed ensemble",
        "sources": source_paths,
        "num_models": len(probabilities),
        "center_metrics_raw": raw_metrics,
        "center_metrics_smoothed": smoothed_metrics,
        "samplewise_metrics_smoothed": sample_metrics,
    }
    save_json(summary, output / "summary.json")
    np.savez_compressed(
        output / "oof_predictions.npz",
        row_indices=np.arange(len(frame)),
        probabilities=ensemble,
        smoothed_probabilities=smoothed,
        targets=targets,
    )
    print(
        f"models={len(probabilities)}; raw center={raw_metrics['macro_f1']:.4%}; "
        f"smoothed center={smoothed_metrics['macro_f1']:.4%}; "
        f"smoothed sample-wise={sample_metrics['macro_f1']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
