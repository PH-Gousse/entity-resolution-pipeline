"""CLI entry point: python -m er"""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="er",
        description="Entity resolution pipeline — blocking, matching, calibration, evaluation.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g., configs/amazon_google.yaml)",
    )
    parser.add_argument(
        "--step",
        choices=["download", "train", "evaluate", "all"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from er.config import load_config  # noqa: I001
    from er.pipeline import (
        load_data, run_blocking, run_evaluate, run_features, run_pipeline, run_train,
    )

    config = load_config(args.config)

    if args.step == "download":
        from er.download import download
        download()
        return 0

    if args.step == "all":
        run_pipeline(config)
        return 0

    # Shared data loading for train / evaluate steps
    df_a, df_b, train, valid, test = load_data(config)
    X_train, y_train, X_valid, y_valid, X_test, y_test = run_features(
        config, df_a, df_b, train, valid, test,
    )

    if args.step == "train":
        _, blocking_ceiling = run_blocking(config, df_a, df_b, train, valid, test)
        run_train(config, X_train, y_train, X_valid, y_valid)
        print(f"Blocking recall ceiling: {blocking_ceiling:.3f}")
        print("Training complete. Run --step evaluate next.")
        return 0

    if args.step == "evaluate":
        from pathlib import Path

        from er.calibrate import calibrate
        from er.train import load_model

        model_path = Path(config.pipeline.artifacts_dir) / "model.txt"
        if not model_path.exists():
            print(f"ERROR: No trained model at {model_path}. Run --step train first.",
                  file=sys.stderr)
            return 1

        booster = load_model(model_path)
        calibrated_model, threshold = calibrate(booster, X_valid, y_valid)
        run_evaluate(config, calibrated_model, threshold, booster,
                     X_test, y_test, X_valid, y_valid)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
