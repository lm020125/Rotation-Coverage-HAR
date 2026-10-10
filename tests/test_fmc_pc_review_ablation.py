from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_realdisp_fmc_pc_review_ablation import (  # noqa: E402
    METHODS,
    SEEDS,
    cached_metrics_match_checkpoint,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_review_matrix_trains_only_missing_empirical_no_pc_cell() -> None:
    assert SEEDS == (256, 3407, 114514)
    assert set(METHODS) == {
        "uniform_no_pc",
        "uniform_pc",
        "empirical_no_pc",
        "empirical_pc",
    }
    assert [name for name, value in METHODS.items() if value["train"]] == [
        "empirical_no_pc"
    ]


def test_cached_metrics_require_matching_path_and_sha256(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen checkpoint")
    metrics = tmp_path / "evaluation" / "metrics.json"
    provenance = tmp_path / "evaluation" / "checkpoint_provenance.json"
    write_json(
        metrics,
        {"evaluation_protocol": {"checkpoint": str(checkpoint.resolve())}},
    )
    write_json(
        provenance,
        {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": digest(checkpoint),
        },
    )
    assert cached_metrics_match_checkpoint(metrics, provenance, checkpoint)

    checkpoint.write_bytes(b"different checkpoint")
    assert not cached_metrics_match_checkpoint(metrics, provenance, checkpoint)


def test_cached_metrics_reject_checkpoint_path_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "expected" / "best.pt"
    other = tmp_path / "other" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    other.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"same bytes")
    other.write_bytes(b"same bytes")
    metrics = tmp_path / "evaluation" / "metrics.json"
    provenance = tmp_path / "evaluation" / "checkpoint_provenance.json"
    write_json(metrics, {"evaluation_protocol": {"checkpoint": str(other)}})
    write_json(
        provenance,
        {
            "checkpoint": str(other),
            "checkpoint_sha256": digest(checkpoint),
        },
    )
    assert not cached_metrics_match_checkpoint(metrics, provenance, checkpoint)


def test_cached_metrics_reject_missing_or_malformed_provenance(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    metrics = tmp_path / "metrics.json"
    provenance = tmp_path / "checkpoint_provenance.json"
    write_json(
        metrics,
        {"evaluation_protocol": {"checkpoint": str(checkpoint.resolve())}},
    )
    assert not cached_metrics_match_checkpoint(metrics, provenance, checkpoint)
    provenance.write_text("not-json", encoding="utf-8")
    assert not cached_metrics_match_checkpoint(metrics, provenance, checkpoint)
