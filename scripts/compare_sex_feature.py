#!/usr/bin/env python3
"""Train and compare XGBoost models with and without the ``sex`` feature."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustworthy_ai.evaluation import evaluate_predictions
from trustworthy_ai.model import train_baseline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, default=Path("data/processed/adult_train.csv"))
    parser.add_argument("--test-input", type=Path, default=Path("data/processed/adult_test.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models/sex_feature_comparison"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def _train_variant(
    *,
    name: str,
    excluded_features: list[str],
    args: argparse.Namespace,
) -> tuple[dict, pd.DataFrame]:
    variant_dir = args.output_dir / name
    metadata = train_baseline(
        train_input=args.train_input,
        test_input=args.test_input,
        output_dir=variant_dir,
        random_state=args.random_state,
        excluded_features=excluded_features,
    )
    predictions = pd.read_csv(metadata.predictions_path)
    metrics = evaluate_predictions(predictions)
    metrics["model"] = name
    metrics["excluded_features"] = excluded_features
    (variant_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics, predictions


def _write_tables(metrics: list[dict], output_dir: Path) -> None:
    performance_rows = []
    fairness_rows = []
    group_rows = []
    for result in metrics:
        model = result["model"]
        performance_rows.append({"model": model, **result["overall"]})
        fairness_rows.append({"model": model, **result["fairness"]})
        for group, values in result["by_group"].items():
            group_rows.append({"model": model, "sex": group, **values})
    pd.DataFrame(performance_rows).to_csv(output_dir / "performance_comparison.csv", index=False)
    pd.DataFrame(fairness_rows).to_csv(output_dir / "fairness_comparison.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output_dir / "group_metrics.csv", index=False)


def _write_figures(metrics: list[dict], output_dir: Path) -> None:
    def _annotate_bars(axis: plt.Axes, *, decimals: int = 4) -> None:
        for bar in axis.patches:
            height = bar.get_height()
            axis.annotate(
                f"{height:.{decimals}f}",
                (bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8,
            )

    labels = [result["model"] for result in metrics]
    performance = pd.DataFrame(
        {
            "Balanced accuracy": [result["overall"]["balanced_accuracy"] for result in metrics],
            "F1": [result["overall"]["f1"] for result in metrics],
            "DPD": [result["fairness"]["demographic_parity_difference"] for result in metrics],
            "EOD": [result["fairness"]["equalized_odds_difference"] for result in metrics],
        },
        index=labels,
    )
    axes = performance.plot.bar(
        subplots=True,
        layout=(2, 2),
        figsize=(8, 6),
        legend=False,
        ylim=(0, 1),
        rot=0,
        title=list(performance.columns),
    )
    for axis in axes.ravel():
        axis.set_ylabel("Value")
        axis.set_xlabel("")
        _annotate_bars(axis)
    plt.tight_layout()
    plt.savefig(output_dir / "performance_fairness_comparison.png", dpi=300)
    plt.close()

    group_rows = []
    for result in metrics:
        for sex, values in result["by_group"].items():
            group_rows.append(
                {
                    "model": result["model"],
                    "sex": sex,
                    "selection_rate": values["selection_rate"],
                }
            )
    group_frame = pd.DataFrame(group_rows)
    chart = group_frame.pivot(index="sex", columns="model", values="selection_rate")
    chart_axes = chart.plot.bar(figsize=(7, 5), ylim=(0, 1), rot=0)
    plt.ylabel("Positive prediction rate")
    plt.xlabel("Sex group")
    _annotate_bars(chart_axes)
    plt.tight_layout()
    plt.savefig(output_dir / "group_positive_prediction_rates.png", dpi=300)
    plt.close()


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        ("with_sex", []),
        ("without_sex", ["sex"]),
    ]
    results = [
        _train_variant(name=name, excluded_features=excluded, args=args)
        for name, excluded in variants
    ]
    metrics = [result[0] for result in results]
    _write_tables(metrics, args.output_dir)
    _write_figures(metrics, args.output_dir)
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
