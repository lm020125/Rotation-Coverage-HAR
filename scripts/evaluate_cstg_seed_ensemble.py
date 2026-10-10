from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_temporal_decoder_cv import evaluate_labels  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.protocol_gate import hard_vote_with_probability_tiebreak  # noqa: E402
from rcbd_har.utils import save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed equal-weight CSTG seed ensemble"
    )
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--tie-probability-inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.inputs) != len(args.tie_probability_inputs):
        raise ValueError("CSTG and tie-probability input counts must match")

    frame = load_manifest(args.manifest).reset_index(drop=True)
    targets = frame.label.to_numpy(np.int64)
    expected_rows = np.arange(len(frame))
    predictions = []
    probability_support = []
    source_paths = []
    tie_paths = []
    for raw_cstg, raw_probability in zip(
        args.inputs, args.tie_probability_inputs, strict=True
    ):
        cstg_path = Path(raw_cstg).resolve()
        probability_path = Path(raw_probability).resolve()
        cstg = np.load(cstg_path / "oof_predictions.npz")
        protocol = np.load(probability_path / "oof_predictions.npz")
        for name, archive in (("CSTG", cstg), ("protocol", protocol)):
            if not np.array_equal(archive["row_indices"], expected_rows):
                raise ValueError(f"{name} OOF rows do not align: {cstg_path}")
            if not np.array_equal(archive["targets"], targets):
                raise ValueError(f"{name} OOF targets do not align: {cstg_path}")
        predictions.append(cstg["gated_predictions"].astype(np.int64))
        probability_support.append(protocol["probabilities"].astype(np.float32))
        source_paths.append(str(cstg_path))
        tie_paths.append(str(probability_path))

    voted = hard_vote_with_probability_tiebreak(
        np.stack(predictions), np.stack(probability_support)
    )
    metrics = evaluate_labels(voted, frame, int(targets.max()) + 1)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "fixed equal hard vote of CSTG seeds",
        "strict_outer_test_label_free": True,
        "weight_selection": "none; all models have fixed equal weight",
        "tie_break": "mean prior-aligned probability restricted to tied classes",
        "sources": source_paths,
        "tie_probability_sources": tie_paths,
        "overall": metrics,
    }
    save_json(summary, output / "summary.json")
    np.savez_compressed(
        output / "oof_predictions.npz",
        row_indices=expected_rows,
        voted_predictions=voted,
        targets=targets,
    )
    print(
        f"models={len(predictions)}; center={metrics['center']['macro_f1']:.4%}; "
        f"sample={metrics['samplewise']['macro_f1']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
