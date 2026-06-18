"""
Pipeline Orchestrator — Coordinate all training, validation, and reporting steps.

Runs the complete publication-grade pipeline:
1. Data loading + barcode deduplication
2. Production XGBoost conformal models
3. GNN (MPNN) training (optional)
4. Selectivity models
5. Y-randomization (all 4 subtypes)
6. SHAP explainability (all 4 subtypes)
7. Evaluation with real conformal intervals
8. External validation (GPCRdb)
9. Literature benchmarking
"""

import sys
import time
import subprocess


def run_step(cmd: list[str], description: str, allow_failure: bool = False):
    """Run a pipeline step as a subprocess."""
    print(f"\n{'=' * 70}")
    print(f"STEP: {description}")
    print(f"CMD:  python {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable] + cmd,
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        status = "FAILED" if not allow_failure else "FAILED (non-critical)"
        print(
            f"\n[{status}] {description} (exit code {result.returncode}) [{elapsed:.1f}s]"
        )
        if not allow_failure:
            print("[ABORTING] Pipeline halted due to critical failure.")
            sys.exit(1)
    else:
        print(f"\n[SUCCESS] {description} [{elapsed:.1f}s]")


def main():
    start_time = time.time()

    print("=" * 70)
    print("ADENOSINE SELECTIVITY MODEL — FULL PUBLICATION PIPELINE")
    print("=" * 70)

    # Parse args
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-retrain", action="store_true", help="Skip production retraining"
    )
    parser.add_argument("--skip-gnn", action="store_true", help="Skip GNN training")
    parser.add_argument(
        "--skip-nested-cv",
        action="store_true",
        help="Skip nested CV (uses default params)",
    )
    parser.add_argument(
        "--y-rand-iterations",
        type=int,
        default=20,
        help="Y-randomization iterations per subtype",
    )
    parser.add_argument(
        "--gnn-epochs", type=int, default=100, help="GNN training epochs"
    )
    args = parser.parse_args()

    # Step 1: Production model retraining with conformal wrapping
    if not args.skip_retrain:
        run_step(
            ["-m", "src.retrain_production"],
            "Production Model Training & Conformal Prediction (MAPIE)",
        )
    else:
        print("[INFO] Skipping production retraining (using existing models).")

    # Step 2: GNN training (optional)
    if not args.skip_gnn:
        run_step(
            ["-m", "src.gnn_model", "--all", "--epochs", str(args.gnn_epochs)],
            f"GNN (MPNN/GINE) Training — All Subtypes ({args.gnn_epochs} epochs)",
            allow_failure=True,
        )

    # Step 3: Pairwise selectivity models
    run_step(
        ["-m", "src.selectivity_models"],
        "Pairwise Affinity Difference Selectivity Models",
    )

    # Step 4: Y-Randomization for ALL subtypes
    run_step(
        [
            "-m",
            "src.y_randomization",
            "--all",
            "--iterations",
            str(args.y_rand_iterations),
        ],
        f"Y-Randomization Robustness Check (ALL Subtypes, n={args.y_rand_iterations})",
    )

    # Step 5: SHAP analysis for ALL subtypes
    run_step(
        ["-m", "src.shap_analysis", "--all"],
        "SHAP Tree Explainability & Chemical Sanity (ALL Subtypes)",
    )

    # Step 6: Evaluation (with and without decoys)
    run_step(
        ["-m", "src.evaluator"],
        "Conformal Model Metrics Evaluator (Full + Actives-Only)",
    )

    # Step 7: External validation
    run_step(
        ["-m", "src.external_validation"],
        "External Validation (GPCRdb Blind Test)",
        allow_failure=True,
    )

    # Step 8: Literature benchmarking
    run_step(
        ["-m", "src.literature_benchmark"],
        "Literature Benchmark Comparison",
        allow_failure=True,
    )

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Total Runtime: {hours}h {minutes}m {seconds}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
