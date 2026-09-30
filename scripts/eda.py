#!/usr/bin/env python3
"""Create exploratory summaries and visualizations for the Adult dataset."""  # noqa: EXE001

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INCOME_LABELS = {0: "<=50K", 1: ">50K"}
INCOME_ORDER = ["<=50K", ">50K"]
COLORS = {"<=50K": "#4C78A8", ">50K": "#F58518"}


def _load_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Clean ETL output not found: {path}")
    frame = pd.read_csv(path)
    required = {"income", "sex", "age", "education", "hours-per-week"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"EDA input is missing required columns: {sorted(missing)}")
    frame["income_label"] = frame["income"].map(INCOME_LABELS)
    if frame["income_label"].isna().any():
        raise ValueError("EDA input contains unknown income codes")
    return frame


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_income_distribution(frame: pd.DataFrame, output_dir: Path) -> None:
    counts = frame["income_label"].value_counts().reindex(INCOME_ORDER)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=[COLORS[label] for label in counts.index])
    ax.bar_label(bars, fmt="{:,.0f}")
    ax.set(title="Income target distribution", xlabel="Annual income", ylabel="Rows")
    _save_figure(fig, output_dir, "income_distribution.png")


def _plot_income_by_sex(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rates = (
        frame.groupby("sex", observed=True)["income"]
        .agg(rows="size", above_50k="mean")
        .sort_values("above_50k", ascending=False)
    )
    rates["above_50k_percent"] = rates["above_50k"] * 100
    rates.reset_index().to_csv(output_dir / "income_rate_by_sex.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_data = rates["above_50k_percent"].sort_values()
    bars = ax.barh(plot_data.index, plot_data.values, color="#59A14F")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set(
        title="Share above $50K by sex",
        xlabel="Rows above $50K (%)",
        ylabel="Sex",
        xlim=(0, max(100, plot_data.max() * 1.2)),
    )
    _save_figure(fig, output_dir, "income_rate_by_sex.png")
    return rates


def _plot_age_by_income(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label in INCOME_ORDER:
        values = frame.loc[frame["income_label"] == label, "age"]
        ax.hist(values, bins=30, alpha=0.6, label=label, color=COLORS[label])
    ax.set(title="Age distribution by income class", xlabel="Age", ylabel="Rows")
    ax.legend(title="Income")
    _save_figure(fig, output_dir, "age_by_income.png")


def _plot_education_by_income(frame: pd.DataFrame, output_dir: Path) -> None:
    order = (
        frame.groupby("education", observed=True)["education-num"]
        .median()
        .sort_values()
        .index
    )
    rates = (
        frame.groupby("education", observed=True)["income"]
        .mean()
        .reindex(order)
        .mul(100)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(rates.index, rates.values, color="#E45756")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set(
        title="Share above $50K by education level",
        xlabel="Rows above $50K (%)",
        ylabel="Education",
        xlim=(0, max(100, rates.max() * 1.2)),
    )
    _save_figure(fig, output_dir, "income_rate_by_education.png")


def _plot_missing_values(frame: pd.DataFrame, output_dir: Path) -> None:
    missing = frame.replace("Missing", pd.NA).isna().sum().sort_values(ascending=True)
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(missing.index, missing.values, color="#B279A2")
    ax.bar_label(bars, fmt="{:,.0f}")
    ax.set(title="Rows with documented missing categorical values", xlabel="Rows", ylabel="Feature")
    _save_figure(fig, output_dir, "missing_values.png")


def _plot_numeric_distributions(frame: pd.DataFrame, output_dir: Path) -> None:
    columns = ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, column in zip(axes.flat, columns):
        ax.hist(frame[column], bins=30, color="#72B7B2", edgecolor="white")
        ax.set(title=column, xlabel=column, ylabel="Rows")
        ax.ticklabel_format(style="plain", axis="x")
    axes.flat[-1].axis("off")
    fig.suptitle("Numeric feature distributions", y=1.02)
    _save_figure(fig, output_dir, "numeric_distributions.png")


def run_eda(
    input_path: str | Path = "data/processed/adult_clean.csv",
    output_dir: str | Path = "data/eda",
) -> dict[str, object]:
    """Generate EDA tables and figures from the cleaned Adult dataset."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_data(input_path)

    _plot_income_distribution(frame, output_dir)
    rates = _plot_income_by_sex(frame, output_dir)
    _plot_age_by_income(frame, output_dir)
    _plot_education_by_income(frame, output_dir)
    _plot_missing_values(frame, output_dir)
    _plot_numeric_distributions(frame, output_dir)

    frame.drop(columns="income_label").describe(include="all").transpose().to_csv(
        output_dir / "descriptive_statistics.csv"
    )
    summary = {
        "input": str(input_path),
        "rows": len(frame),
        "figures": sorted(path.name for path in output_dir.glob("*.png")),
        "income_rate_by_sex_percent": {
            str(sex): round(float(rate), 3)
            for sex, rate in rates["above_50k_percent"].items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        default=Path("data/processed/adult_clean.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/eda"))
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    summary = run_eda(**vars(_build_parser().parse_args(argv)))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
