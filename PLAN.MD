# Fairness Auditing of an Income-Classification Model: SHAP + Proxy-Feature Hunt

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

   [log] Updated EDA outputs with a paper-oriented dataset summary table and relationship-based
   above-$50K rate figure alongside the income distribution and sex-group rate figure.

   [log] Added `scripts/train_xgboost.py` and `trustworthy_ai.model` to fit the baseline XGBoost model with training-only preprocessing on the fixed official split; artifacts and test predictions are written to `data/models/baseline_xgboost/`.
2. **Measure baseline fairness** — Report accuracy, balanced accuracy, precision,
   recall, F1, demographic-parity difference, and equalized-odds difference
   across `sex`.

   [log] Added reusable baseline evaluation metrics and wrote performance and fairness results to `data/models/baseline_xgboost/metrics.json`.

   [log] Exported manuscript-ready baseline performance and group-fairness metrics for the XGBoost comparison.
3. **Compare feature removal** — Evaluate models with `sex` included and removed
   to determine whether excluding the explicit sensitive feature reduces bias.

   [log] Added `scripts/compare_sex_feature.py` to train matched XGBoost models with and without `sex`, write comparison tables to `data/models/sex_feature_comparison/`, and generate the planned performance/fairness and group prediction-rate figures.
4. **Analyze proxy features** — Use TreeSHAP rankings, subgroup comparisons, and
   feature ablation to identify features such as `relationship`,
   `marital-status`, `occupation`, and `hours-per-week` that retain information
   about `sex`.

   [log] Added `scripts/analyze_proxy_features.py` to aggregate TreeSHAP
   contributions by original feature and export subgroup distribution and
   sex-association evidence for the sex-excluded model.
5. **Evaluate proxy removal** — Retrain after removing the strongest suspected
   proxies and measure the fairness, performance, and explainability trade-offs.

   [log] The proxy analysis script now retrains individual and joint ablations
   and exports `proxy_ablation_metrics.csv`, three manuscript-ready figures,
   and an analysis summary under `data/models/proxy_feature_analysis/`.

   [log] Added pre-specified negative controls (`education-num`, `capital-gain`,
   `capital-loss`, and `race`) to the proxy screen; none were marked as
   candidate proxies, and their evidence is exported separately.
6. **Apply fairness mitigation** — Use Fairlearn `ExponentiatedGradient` with
   demographic-parity or equalized-odds constraints, then compare it with the
   unconstrained models.

   [log] Added `scripts/constrain_learning.py` to train Fairlearn
   `ExponentiatedGradient` models with demographic-parity and equalized-odds
   constraints, export comparison metrics and figures, and evaluate the
   resulting fairness--performance trade-offs. output to `data/models/constrained_learning`
   and `data/models/constrained_learning_strict`.
7. **Write the report** — Produce an under-15-page NIPS-style report focused on
   the finding that removing `sex` is insufficient when proxy features remain,
   and on the trade-offs introduced by fairness constraints.

   [log] Drafted the proposed audit method in `report/neurips_2025.tex` and added
   the `Overview of our proposed method` SVG figure at `report/images/overview_proposed_method.svg`.

   [log] Expanded the dataset section with cited explanations of the summary table, income/sex
   figures, and cautious relationship-based proxy motivation.

   [log] Rewrote the evaluation metrics section with explicit BAcc, DPD, and EOD formulas,
   notation for Acc, Pr, Rc, F1, sensitivity, specificity, TPR, FPR, and task-specific
   metric rationale.

   [log] Added the [0,1] value ranges and interpretation of lower versus higher DPD and EOD
   values.

## Scope and limitations

Do not include robustness or threshold analysis. Discuss the dataset's age,
binary treatment of sex, missing values, intersectional groups, limitations of
fairness metrics, and why statistical parity alone does not define fairness.

## References

- [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- [UCI Adult Income Dataset](https://archive.ics.uci.edu/dataset/2/adult)
