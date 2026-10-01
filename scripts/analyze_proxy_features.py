#!/usr/bin/env python3
"""Identify candidate proxies and evaluate controlled proxy-feature ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import DMatrix, XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustworthy_ai.evaluation import evaluate_predictions
from trustworthy_ai.model import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    build_classifier,
    train_baseline,
)


DEFAULT_CANDIDATES = [
    "relationship",
    "marital-status",
    "occupation",
    "hours-per-week",
]
NEGATIVE_CONTROLS = [
    "education-num",
    "capital-gain",
    "capital-loss",
    "race",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, default=Path("data/processed/adult_train.csv"))
    parser.add_argument("--test-input", type=Path, default=Path("data/processed/adult_test.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models/proxy_feature_analysis"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def _cramers_v(feature: pd.Series, sensitive: pd.Series) -> float:
    table = pd.crosstab(feature, sensitive)
    if table.empty or min(table.shape) < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    chi2 = ((observed - expected) ** 2 / expected.clip(min=1e-12)).sum()
    phi2 = chi2 / observed.sum()
    rows, columns = observed.shape
    correction = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / (observed.sum() - 1))
    corrected_rows = rows - ((rows - 1) ** 2) / (observed.sum() - 1)
    corrected_columns = columns - ((columns - 1) ** 2) / (observed.sum() - 1)
    denominator = min(corrected_columns - 1, corrected_rows - 1)
    return float(np.sqrt(correction / denominator)) if denominator > 0 else 0.0


def _distribution_difference(feature: pd.Series, sensitive: pd.Series) -> float:
    groups = sorted(sensitive.dropna().unique())
    if len(groups) != 2:
        return 0.0
    left = feature[sensitive == groups[0]]
    right = feature[sensitive == groups[1]]
    if feature.name in CATEGORICAL_COLUMNS:
        left_rates = left.value_counts(normalize=True)
        right_rates = right.value_counts(normalize=True)
        categories = left_rates.index.union(right_rates.index)
        return float(0.5 * (left_rates.reindex(categories, fill_value=0) -
                            right_rates.reindex(categories, fill_value=0)).abs().sum())
    left_values = left.astype(float)
    right_values = right.astype(float)
    pooled_std = np.sqrt((left_values.var() + right_values.var()) / 2)
    if pooled_std == 0:
        return 0.0
    return float(abs(left_values.mean() - right_values.mean()) / pooled_std)


def _association(feature: pd.Series, sensitive: pd.Series) -> float:
    if feature.name in CATEGORICAL_COLUMNS:
        return _cramers_v(feature, sensitive)
    return float(abs(pd.Series(feature).corr(pd.Series(sensitive), method="spearman")))


def _sex_auc(feature: pd.Series, sensitive: pd.Series) -> float:
    if feature.name in CATEGORICAL_COLUMNS:
        encoded = pd.factorize(feature)[0]
    else:
        encoded = feature.astype(float).to_numpy()
    try:
        score = roc_auc_score(sensitive, encoded)
    except ValueError:
        return 0.5
    return float(max(score, 1 - score))


def _load_model_artifacts(model_dir: Path) -> tuple[object, XGBClassifier]:
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    classifier = build_classifier()
    classifier.load_model(model_dir / "model.json")
    return preprocessor, classifier


def _feature_shap_importance(
    train: pd.DataFrame,
    model_dir: Path,
) -> pd.Series:
    preprocessor, classifier = _load_model_artifacts(model_dir)
    feature_columns = [column for column in FEATURE_COLUMNS if column != "sex"]
    transformed = preprocessor.transform(train[feature_columns])
    contributions = classifier.get_booster().predict(
        DMatrix(transformed, feature_names=preprocessor.get_feature_names_out().tolist()),
        pred_contribs=True,
    )
    transformed_names = preprocessor.get_feature_names_out()
    mean_abs = np.abs(contributions[:, :-1]).mean(axis=0)
    transformed_importance = pd.Series(mean_abs, index=transformed_names)
    original_importance = {}
    for feature in feature_columns:
        matching = transformed_importance[
            transformed_importance.index == feature
        ]
        if matching.empty:
            matching = transformed_importance[
                transformed_importance.index.str.startswith(f"{feature}_")
            ]
        original_importance[feature] = float(matching.sum())
    return pd.Series(original_importance, name="mean_abs_shap")


def _write_proxy_evidence(train: pd.DataFrame, model_dir: Path, output_dir: Path) -> pd.DataFrame:
    sensitive = (train["sex"] == "Male").astype(int)
    shap = _feature_shap_importance(train, model_dir)
    rows = []
    for feature in [column for column in FEATURE_COLUMNS if column != "sex"]:
        series = train[feature]
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": shap[feature],
                "shap_rank": int(shap.rank(ascending=False, method="min")[feature]),
                "distribution_difference": _distribution_difference(series, train["sex"]),
                "sex_association": _association(series, sensitive),
                "sex_auc_single_feature": _sex_auc(series, sensitive),
            }
        )
    evidence = pd.DataFrame(rows).sort_values(
        ["mean_abs_shap", "sex_association"], ascending=False
    )
    evidence["model_reliance_rank"] = evidence["mean_abs_shap"].rank(
        ascending=False, method="min"
    ).astype(int)
    evidence["association_rank"] = evidence["sex_association"].rank(
        ascending=False, method="min"
    ).astype(int)
    evidence["candidate_proxy"] = (
        (evidence["model_reliance_rank"] <= len(evidence) / 2)
        & (evidence["association_rank"] <= len(evidence) / 2)
    )
    evidence["screening_group"] = np.where(
        evidence["feature"].isin(DEFAULT_CANDIDATES),
        "human_motivated_candidate",
        np.where(evidence["feature"].isin(NEGATIVE_CONTROLS), "negative_control", "other"),
    )
    evidence.to_csv(output_dir / "proxy_feature_evidence.csv", index=False)
    evidence[evidence["screening_group"] == "negative_control"].to_csv(
        output_dir / "negative_control_evidence.csv", index=False
    )
    return evidence


def _train_and_evaluate(
    *,
    name: str,
    excluded_features: list[str],
    args: argparse.Namespace,
) -> dict:
    variant_dir = args.output_dir / "models" / name
    metadata = train_baseline(
        train_input=args.train_input,
        test_input=args.test_input,
        output_dir=variant_dir,
        random_state=args.random_state,
        excluded_features=excluded_features,
    )
    predictions = pd.read_csv(metadata.predictions_path)
    metrics = evaluate_predictions(predictions)
    result = {
        "model": name,
        "excluded_features": excluded_features,
        **metrics["overall"],
        **metrics["fairness"],
    }
    for group, values in metrics["by_group"].items():
        result[f"{group}_selection_rate"] = values["selection_rate"]
        result[f"{group}_tpr"] = values["true_positive_rate"]
        result[f"{group}_fpr"] = values["false_positive_rate"]
    return result


def _write_distribution_figure(train: pd.DataFrame, evidence: pd.DataFrame, output_dir: Path) -> None:
    selected = [
        feature for feature in DEFAULT_CANDIDATES if feature in train.columns
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, feature in zip(axes.ravel(), selected):
        if feature in CATEGORICAL_COLUMNS:
            rates = (
                train.groupby([feature, "sex"], observed=True)
                .size()
                .groupby(level=1)
                .transform(lambda values: values / values.sum())
            )
            frame = rates.rename("rate").reset_index()
            chart = frame.pivot(index=feature, columns="sex", values="rate")
            chart.plot.bar(ax=axis, rot=45)
            axis.set_ylabel("Group proportion")
        else:
            for group, group_frame in train.groupby("sex", observed=True):
                axis.hist(
                    group_frame[feature],
                    bins=25,
                    alpha=0.55,
                    density=True,
                    label=str(group),
                )
            axis.set_ylabel("Density")
        axis.set_title(feature)
        axis.set_xlabel("")
        axis.legend(title="sex")
    figure.suptitle("Candidate proxy feature distributions by sex group")
    figure.tight_layout()
    figure.savefig(output_dir / "proxy_feature_distributions.png", dpi=300)
    plt.close(figure)


def _write_shap_figure(evidence: pd.DataFrame, output_dir: Path) -> None:
    frame = evidence.sort_values("mean_abs_shap")
    colors = ["tab:orange" if candidate else "tab:blue"
              for candidate in frame["candidate_proxy"]]
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(frame["feature"], frame["mean_abs_shap"], color=colors)
    axis.set_xlabel("Mean absolute TreeSHAP contribution")
    axis.set_ylabel("")
    axis.set_title("TreeSHAP importance for the sex-excluded model")
    figure.tight_layout()
    figure.savefig(output_dir / "proxy_feature_shap_importance.png", dpi=300)
    plt.close(figure)


def _write_tradeoff_figure(results: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(
        results["balanced_accuracy"],
        results["demographic_parity_difference"],
        s=70,
        color="tab:blue",
    )
    for _, row in results.iterrows():
        axis.annotate(row["model"], (row["balanced_accuracy"], row["demographic_parity_difference"]),
                      xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Balanced accuracy")
    axis.set_ylabel("Demographic-parity difference")
    axis.set_title("Performance–fairness trade-off across proxy ablations")
    figure.tight_layout()
    figure.savefig(output_dir / "proxy_ablation_tradeoff.png", dpi=300)
    plt.close(figure)


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train_input)

    without_sex_dir = args.output_dir / "models" / "without_sex"
    train_baseline(
        train_input=args.train_input,
        test_input=args.test_input,
        output_dir=without_sex_dir,
        random_state=args.random_state,
        excluded_features=["sex"],
    )
    evidence = _write_proxy_evidence(train, without_sex_dir, args.output_dir)
    selected = [
        feature for feature in DEFAULT_CANDIDATES
        if bool(evidence.loc[evidence["feature"] == feature, "candidate_proxy"].iloc[0])
    ]
    if not selected:
        selected = [
            feature for feature in DEFAULT_CANDIDATES
            if feature in set(evidence.head(4)["feature"])
        ]
    variants = [
        ("with_sex", []),
        ("without_sex", ["sex"]),
    ]
    variants.extend(
        (f"without_{feature}", ["sex", feature])
        for feature in selected
    )
    variants.append(("without_selected_proxies", ["sex", *selected]))
    results = [_train_and_evaluate(name=name, excluded_features=excluded, args=args)
               for name, excluded in variants]
    results_frame = pd.DataFrame(results)
    results_frame.to_csv(args.output_dir / "proxy_ablation_metrics.csv", index=False)
    _write_shap_figure(evidence, args.output_dir)
    _write_distribution_figure(train, evidence, args.output_dir)
    _write_tradeoff_figure(results_frame, args.output_dir)
    summary = {
        "selected_proxy_features": selected,
        "negative_control_features": NEGATIVE_CONTROLS,
        "negative_controls_marked_as_proxies": evidence.loc[
            evidence["feature"].isin(NEGATIVE_CONTROLS) & evidence["candidate_proxy"],
            "feature",
        ].tolist(),
        "candidate_rule": "top half by mean absolute TreeSHAP and top half by sex association",
        "artifacts": {
            "evidence_table": str(args.output_dir / "proxy_feature_evidence.csv"),
            "ablation_table": str(args.output_dir / "proxy_ablation_metrics.csv"),
            "shap_figure": str(args.output_dir / "proxy_feature_shap_importance.png"),
            "distribution_figure": str(args.output_dir / "proxy_feature_distributions.png"),
            "tradeoff_figure": str(args.output_dir / "proxy_ablation_tradeoff.png"),
        },
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
