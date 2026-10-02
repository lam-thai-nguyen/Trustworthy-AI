#!/usr/bin/env python3
"""Train and evaluate Fairlearn fairness-constrained income classifiers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustworthy_ai.evaluation import evaluate_predictions
from trustworthy_ai.model import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    build_classifier,
    build_preprocessor,
)


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
        default=Path("data/models/constrained_learning"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=50)
    return parser


def _load_split(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Clean dataset split not found: {path}")
    frame = pd.read_csv(path)
    required = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS) | {TARGET_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"Input contains null values: {path}")
    return frame


def _build_base_estimator(random_state: int) -> Pipeline:
    feature_columns = [
        column for column in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS if column != "sex"
    ]
    categorical_columns = [
        column for column in CATEGORICAL_COLUMNS if column != "sex"
    ]
    numeric_columns = list(NUMERIC_COLUMNS)
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    categorical_columns=categorical_columns,
                    numeric_columns=numeric_columns,
                ),
            ),
            ("classifier", build_classifier(random_state)),
        ]
    )


def _evaluate_model(
    *,
    name: str,
    constraint,
    train: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    features = [column for column in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS if column != "sex"]
    estimator = ExponentiatedGradient(
        _build_base_estimator(args.random_state),
        constraints=constraint,
        eps=args.eps,
        max_iter=args.max_iter,
        nu=None,
        sample_weight_name="classifier__sample_weight",
    )
    estimator.fit(
        train[features],
        train[TARGET_COLUMN],
        sensitive_features=train["sex"],
    )
    predictions = estimator.predict(test[features]).astype("int8")
    prediction_frame = test[["sex", TARGET_COLUMN]].copy()
    prediction_frame["prediction"] = predictions

    model_dir = args.output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(model_dir / "test_predictions.csv", index=False)
    joblib.dump(estimator, model_dir / "model.joblib")

    metrics = evaluate_predictions(prediction_frame)
    result = {
        "model": name,
        "constraint": type(constraint).__name__,
        "eps": args.eps,
        "max_iter": args.max_iter,
        **metrics["overall"],
        **metrics["fairness"],
    }
    for group, values in metrics["by_group"].items():
        result[f"{group}_selection_rate"] = values["selection_rate"]
        result[f"{group}_tpr"] = values["true_positive_rate"]
        result[f"{group}_fpr"] = values["false_positive_rate"]
    (model_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _load_reference_results() -> pd.DataFrame:
    rows = []
    ablation_path = Path("data/models/proxy_feature_analysis/proxy_ablation_metrics.csv")
    if ablation_path.is_file():
        ablations = pd.read_csv(ablation_path)
        reference_names = {"without_sex", "without_selected_proxies"}
        rows.extend(
            ablations[ablations["model"].isin(reference_names)].to_dict("records")
        )
    return pd.DataFrame(rows)


def _write_outputs(results: list[dict], output_dir: Path) -> None:
    constrained = pd.DataFrame(results)
    references = _load_reference_results()
    combined = pd.concat([references, constrained], ignore_index=True, sort=False)
    combined.to_csv(output_dir / "constrained_learning_metrics.csv", index=False)
    (output_dir / "constrained_learning_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    labels = {
        "without_sex": "Without sex",
        "without_selected_proxies": "Without selected proxies",
        "demographic_parity": "Demographic parity",
        "equalized_odds": "Equalized odds",
    }
    plot_frame = combined[combined["model"].isin(labels)].copy()
    plot_frame["label"] = plot_frame["model"].map(labels)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, fairness_metric, title in (
        (
            axes[0],
            "demographic_parity_difference",
            "Balanced accuracy vs. demographic-parity difference",
        ),
        (
            axes[1],
            "equalized_odds_difference",
            "Balanced accuracy vs. equalized-odds difference",
        ),
    ):
        for _, row in plot_frame.iterrows():
            axis.scatter(row["balanced_accuracy"], row[fairness_metric], s=65)
            axis.annotate(
                row["label"],
                (row["balanced_accuracy"], row[fairness_metric]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel("Balanced accuracy")
        axis.set_ylabel(fairness_metric.replace("_", " ").title())
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "constrained_learning_tradeoff.png", dpi=300)
    plt.close(figure)

    group_rows = []
    for _, row in plot_frame.iterrows():
        for group in ("Female", "Male"):
            group_rows.append(
                {
                    "model": row["model"],
                    "label": row["label"],
                    "sex": group,
                    "selection_rate": row.get(f"{group}_selection_rate"),
                    "tpr": row.get(f"{group}_tpr"),
                    "fpr": row.get(f"{group}_fpr"),
                }
            )
    group_frame = pd.DataFrame(group_rows)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, metric, title in (
        (axes[0], "selection_rate", "Positive prediction rate"),
        (axes[1], "tpr", "True-positive rate"),
        (axes[2], "fpr", "False-positive rate"),
    ):
        chart = group_frame.pivot(index="label", columns="sex", values=metric)
        chart.plot.bar(ax=axis, ylim=(0, 1), rot=45)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("Rate")
        axis.legend(title="Sex")
    figure.tight_layout()
    figure.savefig(output_dir / "constrained_learning_group_rates.png", dpi=300)
    plt.close(figure)


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = _load_split(args.train_input)
    test = _load_split(args.test_input)

    results = [
        _evaluate_model(
            name="demographic_parity",
            constraint=DemographicParity(),
            train=train,
            test=test,
            args=args,
        ),
        _evaluate_model(
            name="equalized_odds",
            constraint=EqualizedOdds(),
            train=train,
            test=test,
            args=args,
        ),
    ]
    _write_outputs(results, args.output_dir)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
