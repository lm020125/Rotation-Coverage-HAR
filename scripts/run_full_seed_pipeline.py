from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_stage(name: str, command: list[str], summary: Path, force: bool) -> None:
    if summary.exists() and not force:
        print(f"skip {name}: {summary} exists", flush=True)
        return
    print(f"start {name}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not summary.exists():
        raise RuntimeError(f"{name} finished without writing {summary}")
    print(f"complete {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete nested WEAR innovation pipeline for multiple seeds"
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    for seed in args.seeds:
        base = ROOT / "outputs" / f"wear_ul_catboost_seed{seed}"
        residual = ROOT / "outputs" / f"wear_null_residual_meta_seed{seed}"
        prior = ROOT / "outputs" / f"wear_prior_alignment_seed{seed}"
        protocol = ROOT / "outputs" / f"wear_protocol_decoder_dense_seed{seed}"
        if not (base / "summary.json").exists() or not (base / "oof_predictions.npz").exists():
            raise FileNotFoundError(f"Complete base UL output is missing for seed {seed}: {base}")

        run_stage(
            f"seed {seed} null residual",
            [
                python,
                "-u",
                "scripts/run_null_residual_cv.py",
                "--base-output",
                str(base),
                "--output",
                str(residual),
                "--seed",
                str(seed),
                "--meta-select",
            ],
            residual / "summary.json",
            args.force,
        )
        run_stage(
            f"seed {seed} prior alignment",
            [
                python,
                "-u",
                "scripts/run_prior_alignment_cv.py",
                "--source-output",
                str(residual),
                "--inner-root",
                str(residual),
                "--output",
                str(prior),
            ],
            prior / "summary.json",
            args.force,
        )
        run_stage(
            f"seed {seed} dense protocol decoder",
            [
                python,
                "-u",
                "scripts/run_protocol_decoder_cv.py",
                "--source-output",
                str(prior),
                "--inner-root",
                str(residual),
                "--output",
                str(protocol),
                "--strengths",
                "2",
                "2.25",
                "2.5",
                "2.75",
                "3",
                "--minimum-counts",
                "100",
                "120",
                "130",
                "140",
                "150",
                "160",
                "175",
            ],
            protocol / "summary.json",
            args.force,
        )


if __name__ == "__main__":
    main()
