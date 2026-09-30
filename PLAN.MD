# Fairness Auditing of a Credit Model: SHAP + Proxy-Feature Hunt

## Objective

Show that removing `sex` from a credit-income model does not necessarily remove
bias because proxy features can preserve sensitive information.

## Dataset

Use the [UCI Adult Income dataset](https://archive.ics.uci.edu/dataset/2/adult):

- 48,842 instances and 14 categorical/integer features
- Target: income over $50,000 per year
- Handle the documented missing values

## Implementation plan

1. **Build a reproducible pipeline** — Load the dataset from `data/adult` and clean it, create a
   fixed train/test split, and train the baseline XGBoost model.

   [log] From the repository root, run `python3 scripts/run_etl.py`. It writes cleaned data and
   run metadata to `data/processed/`.

   [log] Run `python3 scripts/eda.py` to create exploratory summary tables and visualizations in
   `data/eda/`.
2. **Measure baseline fairness** — Report accuracy, balanced accuracy, precision,
   recall, F1, demographic-parity difference, and equalized-odds difference
   across `sex`.
3. **Compare feature removal** — Evaluate models with `sex` included and removed
   to determine whether excluding the explicit sensitive feature reduces bias.
4. **Analyze proxy features** — Use TreeSHAP rankings, subgroup comparisons, and
   feature ablation to identify features such as `relationship`,
   `marital-status`, `occupation`, and `hours-per-week` that retain information
   about `sex`.
5. **Evaluate proxy removal** — Retrain after removing the strongest suspected
   proxies and measure the fairness, performance, and explainability trade-offs.
6. **Apply fairness mitigation** — Use Fairlearn `ExponentiatedGradient` with
   demographic-parity or equalized-odds constraints, then compare it with the
   unconstrained models.
7. **Summarize model results** — Create a comparison table covering all model
   variants and the selected performance and fairness metrics.
8. **Write the report** — Produce an under-15-page NIPS-style report focused on
   the finding that removing `sex` is insufficient when proxy features remain,
   and on the trade-offs introduced by fairness constraints.

   [log] Drafted the proposed audit method in `report/neurips_2025.tex` and added
   the `Overview of our proposed method` SVG figure at `report/images/overview_proposed_method.svg`.

## Scope and limitations

Do not include robustness or threshold analysis. Discuss the dataset's age,
binary treatment of sex, missing values, intersectional groups, limitations of
fairness metrics, and why statistical parity alone does not define fairness.

## References

- [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- [UCI Adult Income Dataset](https://archive.ics.uci.edu/dataset/2/adult)
