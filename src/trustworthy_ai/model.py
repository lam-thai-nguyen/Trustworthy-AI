"""Reusable preprocessing and baseline XGBoost training utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from trustworthy_ai.etl import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

TARGET_COLUMN = "income"
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


@dataclass(frozen=True)
class TrainingMetadata:
    """Configuration and output summary for one baseline training run."""

    train_input: str
    test_input: str
    output_dir: str
    feature_columns: list[str]
    target_column: str
    categorical_columns: list[str]
    numeric_columns: list[str]
    train_rows: int
    test_rows: int
    positive_train_rows: int
    positive_test_rows: int
    transformed_feature_count: int
    random_state: int
    model_parameters: dict[str, Any]
    model_path: str
    preprocessor_path: str
    predictions_path: str


def _load_split(path: Path) -> pd.DataFrame:
    """Load and validate one cleaned Adult split."""
    if not path.is_file():
        raise FileNotFoundError(f"Clean dataset split not found: {path}")
    frame = pd.read_csv(path)
    required = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training input is missing required columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"Training input contains null values: {path}")
    unknown_targets = sorted(set(frame[TARGET_COLUMN]) - {0, 1})
    if unknown_targets:
        raise ValueError(f"Training input contains unknown target values: {unknown_targets}")
    return frame


def build_preprocessor() -> ColumnTransformer:
    """Build a leakage-safe transformer for the Adult feature schema."""
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_COLUMNS),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_classifier(random_state: int = 42) -> XGBClassifier:
    """Build the deterministic baseline classifier."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=random_state,
        n_jobs=1,
        tree_method="hist",
    )


def train_baseline(
    train_input: str | Path = "data/processed/adult_train.csv",
    test_input: str | Path = "data/processed/adult_test.csv",
    output_dir: str | Path = "data/models/baseline_xgboost",
    random_state: int = 42,
) -> TrainingMetadata:
    """Fit the baseline model on the official fixed Adult train/test split."""
    train_path = Path(train_input)
    test_path = Path(test_input)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train = _load_split(train_path)
    test = _load_split(test_path)
    x_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    x_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    preprocessor = build_preprocessor()
    classifier = build_classifier(random_state)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(x_train, y_train)

    transformed_feature_count = len(
        pipeline.named_steps["preprocessor"].get_feature_names_out()
    )
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype("int8")
    prediction_frame = test[["sex", TARGET_COLUMN]].copy()
    prediction_frame["probability"] = probabilities
    prediction_frame["prediction"] = predictions
    predictions_path = output_path / "test_predictions.csv"
    prediction_frame.to_csv(predictions_path, index=False)

    model_path = output_path / "model.json"
    pipeline.named_steps["classifier"].save_model(model_path)
    preprocessor_path = output_path / "preprocessor.joblib"
    joblib.dump(pipeline.named_steps["preprocessor"], preprocessor_path)

    model_parameters = {
        key: value
        for key, value in classifier.get_params().items()
        if key in {
            "objective",
            "eval_metric",
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "random_state",
            "n_jobs",
            "tree_method",
        }
    }
    metadata = TrainingMetadata(
        train_input=str(train_path),
        test_input=str(test_path),
        output_dir=str(output_path),
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        categorical_columns=CATEGORICAL_COLUMNS,
        numeric_columns=NUMERIC_COLUMNS,
        train_rows=len(train),
        test_rows=len(test),
        positive_train_rows=int(y_train.sum()),
        positive_test_rows=int(y_test.sum()),
        transformed_feature_count=transformed_feature_count,
        random_state=random_state,
        model_parameters=model_parameters,
        model_path=str(model_path),
        preprocessor_path=str(preprocessor_path),
        predictions_path=str(predictions_path),
    )
    (output_path / "metadata.json").write_text(
        json.dumps(asdict(metadata), indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata
