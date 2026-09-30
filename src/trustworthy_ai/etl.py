"""ETL for the UCI Adult income dataset.

The dataset is distributed as two comma-separated files with inconsistent
target formatting: ``adult.test`` has a leading metadata line and periods
after its target labels. This module produces a consistent, validated table
without dropping rows with documented missing values.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable  # noqa: UP035

import pandas as pd

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
CATEGORICAL_COLUMNS = [column for column in COLUMNS if column not in NUMERIC_COLUMNS + ["income"]]
MISSING_VALUE = "Missing"
INCOME_LABELS = {"<=50K": 0, ">50K": 1}


@dataclass(frozen=True)
class ETLMetadata:
    """Summary of one deterministic ETL run."""

    source_train: str
    source_test: str
    output_dir: str
    train_rows: int
    test_rows: int
    total_rows: int
    missing_values_replaced: int
    missing_values_by_column: dict[str, int]
    income_counts: dict[str, int]
    columns: list[str]


def _read_split(path: Path, *, is_test: bool) -> pd.DataFrame:
    """Read and normalize one raw Adult split."""
    skiprows = 1 if is_test else 0
    frame = pd.read_csv(
        path,
        names=COLUMNS,
        skiprows=skiprows,
        header=None,
        skipinitialspace=True,
        na_filter=False,
        dtype="string",
    )
    if frame.empty:
        raise ValueError(f"Dataset split is empty: {path}")

    for column in COLUMNS:
        frame[column] = frame[column].str.strip()

    # The official test file writes labels as ">50K." and "<=50K.".
    frame["income"] = frame["income"].str.removesuffix(".")
    return frame


def _validate_raw(frame: pd.DataFrame, source: Path) -> None:
    """Validate raw values before type conversion."""
    unexpected_income = sorted(set(frame["income"]) - set(INCOME_LABELS))
    if unexpected_income:
        raise ValueError(f"Unexpected income labels in {source}: {unexpected_income}")

    for column in NUMERIC_COLUMNS:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any():
            bad_values = sorted(set(frame.loc[numeric.isna(), column]))
            raise ValueError(f"Non-numeric values in {source} column {column}: {bad_values}")


def _clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert numeric columns and make missing categories explicit."""
    missing_by_column = {
        column: int((frame[column] == "?").sum())
        for column in COLUMNS
        if (frame[column] == "?").any()
    }
    cleaned = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].replace("?", MISSING_VALUE)
    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise").astype("int64")
    cleaned["income"] = cleaned["income"].map(INCOME_LABELS).astype("int8")
    return cleaned, missing_by_column


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_etl(
    data_dir: str | Path = "data/adult",
    output_dir: str | Path = "data/processed",
) -> ETLMetadata:
    """Run the reproducible ETL using the official Adult train/test split."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset file not found: {path}")

    raw_train = _read_split(train_path, is_test=False)
    raw_test = _read_split(test_path, is_test=True)
    _validate_raw(raw_train, train_path)
    _validate_raw(raw_test, test_path)

    train, train_missing = _clean(raw_train)
    test, test_missing = _clean(raw_test)
    combined = pd.concat([train, test], ignore_index=True)
    missing_by_column = {
        column: train_missing.get(column, 0) + test_missing.get(column, 0)
        for column in set(train_missing) | set(test_missing)
    }

    _write_csv(train, output_dir / "adult_train.csv")
    _write_csv(test, output_dir / "adult_test.csv")
    _write_csv(combined, output_dir / "adult_clean.csv")

    metadata = ETLMetadata(
        source_train=str(train_path),
        source_test=str(test_path),
        output_dir=str(output_dir),
        train_rows=len(train),
        test_rows=len(test),
        total_rows=len(combined),
        missing_values_replaced=sum(missing_by_column.values()),
        missing_values_by_column=dict(sorted(missing_by_column.items())),
        income_counts={
            str(label): int(count)
            for label, count in raw_train["income"].value_counts().sort_index().items()
        },
        columns=COLUMNS,
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(asdict(metadata), indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/adult"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    metadata = run_etl(**vars(_build_parser().parse_args(argv)))
    print(json.dumps(asdict(metadata), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
