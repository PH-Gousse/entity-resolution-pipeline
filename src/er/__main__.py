"""CLI entry point: python -m er"""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()

    from er.config import load_config

    config = load_config(args.config)
    print(f"Loaded config: {config.dataset.name}")
    print(f"Pipeline step: {args.step}")

    # TODO: wire up pipeline steps once modules are implemented
    print("Pipeline modules not yet implemented. Scaffold is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
