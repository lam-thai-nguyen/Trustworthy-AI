#!/usr/bin/env python3
"""Evaluate baseline performance and fairness on the Adult test split."""  # noqa: EXE001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

# Make the src-layout package importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from trustworthy_ai.evaluation import evaluate_predictions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/models/baseline_xgboost/test_predictions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/baseline_xgboost/metrics.json"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(f"Prediction input not found: {args.input}")
    metrics = evaluate_predictions(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
