#!/usr/bin/env python3
"""Train the baseline XGBoost model on the cleaned Adult dataset."""  # noqa: EXE001

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

# Make the src-layout package importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustworthy_ai.model import train_baseline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-input",
        type=Path,
        default=Path("data/processed/adult_train.csv"),
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        default=Path("data/processed/adult_test.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models/baseline_xgboost"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    metadata = train_baseline(**vars(_build_parser().parse_args(argv)))
    print(json.dumps(asdict(metadata), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
