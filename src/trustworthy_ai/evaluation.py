"""Performance and group fairness metrics for binary predictions."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_predictions(
    frame: pd.DataFrame,
    *,
    sensitive_column: str = "sex",
    target_column: str = "income",
    prediction_column: str = "prediction",
) -> dict[str, Any]:
    """Evaluate binary predictions overall and across sensitive groups.

    Demographic-parity difference is the largest pairwise difference in
    positive prediction rates. Equalized-odds difference is the larger of the
    largest pairwise TPR difference and largest pairwise FPR difference.
    """
    required = {sensitive_column, target_column, prediction_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction input is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Prediction input is empty")

    y_true = frame[target_column].astype(int)
    y_pred = frame[prediction_column].astype(int)
    if not set(y_true).issubset({0, 1}) or not set(y_pred).issubset({0, 1}):
        raise ValueError("Targets and predictions must contain only binary values 0 and 1")

    overall = {
        "rows": int(len(frame)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    group_metrics: dict[str, dict[str, float | int]] = {}
    for group, group_frame in frame.groupby(sensitive_column, sort=True, observed=True):
        group_true = group_frame[target_column].astype(int)
        group_pred = group_frame[prediction_column].astype(int)
        true_positive = int(((group_true == 1) & (group_pred == 1)).sum())
        false_positive = int(((group_true == 0) & (group_pred == 1)).sum())
        actual_positive = int((group_true == 1).sum())
        actual_negative = int((group_true == 0).sum())
        group_metrics[str(group)] = {
            "rows": int(len(group_frame)),
            "selection_rate": float(group_pred.mean()),
            "true_positive_rate": _rate(true_positive, actual_positive),
            "false_positive_rate": _rate(false_positive, actual_negative),
            "precision": float(precision_score(group_true, group_pred, zero_division=0)),
            "recall": float(recall_score(group_true, group_pred, zero_division=0)),
        }

    if len(group_metrics) < 2:
        raise ValueError("At least two sensitive groups are required for fairness metrics")

    pairs = list(combinations(group_metrics.values(), 2))
    demographic_parity_difference = max(
        abs(left["selection_rate"] - right["selection_rate"]) for left, right in pairs
    )
    true_positive_rate_difference = max(
        abs(left["true_positive_rate"] - right["true_positive_rate"]) for left, right in pairs
    )
    false_positive_rate_difference = max(
        abs(left["false_positive_rate"] - right["false_positive_rate"]) for left, right in pairs
    )
    fairness = {
        "demographic_parity_difference": float(demographic_parity_difference),
        "equalized_odds_difference": float(
            max(true_positive_rate_difference, false_positive_rate_difference)
        ),
        "true_positive_rate_difference": float(true_positive_rate_difference),
        "false_positive_rate_difference": float(false_positive_rate_difference),
    }
    return {
        "sensitive_column": sensitive_column,
        "target_column": target_column,
        "prediction_column": prediction_column,
        "overall": overall,
        "by_group": group_metrics,
        "fairness": fairness,
    }
