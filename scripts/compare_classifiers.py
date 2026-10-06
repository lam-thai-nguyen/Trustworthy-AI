#!/usr/bin/env python3
"""Compare CPU-friendly classifiers on the UCI Adult dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustworthy_ai.etl import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from trustworthy_ai.evaluation import evaluate_predictions
from trustworthy_ai.model import FEATURE_COLUMNS, TARGET_COLUMN


def _load_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    return frame


def _build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    categorical = [column for column in CATEGORICAL_COLUMNS if column in feature_columns]
    numeric = [column for column in NUMERIC_COLUMNS if column in feature_columns]
    return ColumnTransformer(
        [
            ("numeric", "passthrough", numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _build_models(random_state: int) -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=300, solver="liblinear", random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=2, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            n_jobs=1,
            random_state=random_state,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.1,
            max_leaf_nodes=31,
            random_state=random_state,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            loss_function="Logloss",
            verbose=False,
            thread_count=1,
            random_seed=random_state,
            allow_writing_files=False,
        ),
        "Support Vector Machine": LinearSVC(
            C=1.0, max_iter=3000, random_state=random_state
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=1),
        "XGBoost": XGBClassifier(
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
        ),
    }


def _scores(model: object, features: object) -> object:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def _train_one(
    name: str,
    estimator: object,
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
) -> dict:
    start = time.perf_counter()
    transformed_train = preprocessor.fit_transform(x_train)
    transformed_test = preprocessor.transform(x_test)
    if name in {"HistGradientBoosting", "CatBoost"}:
        transformed_train = transformed_train.toarray()
        transformed_test = transformed_test.toarray()
    estimator.fit(transformed_train, y_train)
    scores = _scores(estimator, transformed_test)
    predictions = (scores >= 0.5).astype("int8") if name not in {
        "Support Vector Machine"
    } else (scores >= 0).astype("int8")
    prediction_frame = test[["sex", TARGET_COLUMN]].copy()
    prediction_frame["score"] = scores
    prediction_frame["prediction"] = predictions
    model_dir = output_dir / name.lower().replace(" ", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(model_dir / "test_predictions.csv", index=False)
    joblib.dump(
        Pipeline([("preprocessor", preprocessor), ("classifier", estimator)]),
        model_dir / "model.joblib",
    )
    metrics = evaluate_predictions(prediction_frame)
    metrics["model"] = name
    metrics["train_seconds"] = round(time.perf_counter() - start, 3)
    metrics["transformed_feature_count"] = int(transformed_train.shape[1])
    (model_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-input", type=Path, default=Path("data/processed/adult_train.csv")
    )
    parser.add_argument(
        "--test-input", type=Path, default=Path("data/processed/adult_test.csv")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models/classifier_comparison"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--include-sex",
        action="store_true",
        help="Use sex as a predictive feature; it remains available for evaluation by default.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    train = _load_split(args.train_input)
    test = _load_split(args.test_input)
    excluded = [] if args.include_sex else ["sex"]
    feature_columns = [column for column in FEATURE_COLUMNS if column not in excluded]
    x_train, y_train = train[feature_columns], train[TARGET_COLUMN]
    x_test = test[feature_columns]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, estimator in _build_models(args.random_state).items():
        preprocessor = _build_preprocessor(feature_columns)
        results.append(
            _train_one(
                name,
                estimator,
                preprocessor,
                x_train,
                y_train,
                x_test,
                test,
                args.output_dir,
            )
        )

    rows = []
    for result in results:
        rows.append(
            {
                "model": result["model"],
                **result["overall"],
                **result["fairness"],
                "train_seconds": result["train_seconds"],
            }
        )
    comparison = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "f1"], ascending=False
    )
    comparison.to_csv(args.output_dir / "classifier_comparison.csv", index=False)
    (args.output_dir / "classifier_comparison.json").write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "excluded_features": excluded,
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
