# Copilot Instructions

## Response format

Begin every response after a line containing exactly 20 `=` characters.

## Environment setup

Run python code for this project inside a conda enviroment

```bash
conda create -n trustworthy python=3.14.7
conda activate trustworthy
python -m pip install -r requirements.txt
```

The `python3` interpreter is `/home/thai/miniconda3/envs/trustworthy/bin/python3`. 
If not found, use the interpreter in the output of this bash command `which python3` inside the `trustworthy` environment.

To install new packages for the `trustworthy` environment, use `pip3 install <package>`.
After installing, run `pip3 freeze > requirements.txt`.

## Project

This project audits fairness in an income-classification model. Its central
question is whether removing `sex` prevents bias when other features act as
proxies for it.

Use the UCI Adult dataset stored locally in `data/adult/`. The target is
whether income exceeds `$50K`. Treat `sex` as the sensitive attribute.

## Required workflow

- Keep the pipeline reproducible with a fixed train/test split and documented
  preprocessing.
- Handle the dataset's missing values explicitly.
- Train and compare:
  - a baseline model with `sex`;
  - a model without `sex`;
  - models with suspected proxy features removed;
  - a Fairlearn `ExponentiatedGradient` mitigation model.
- Use XGBoost for the predictive model and TreeSHAP for feature explanations.
- Investigate `relationship`, `marital-status`, `occupation`, and
  `hours-per-week` as potential proxies using SHAP rankings, subgroup
  comparisons, and feature ablation.
- Measure accuracy, balanced accuracy, precision, recall, F1,
  demographic-parity difference, and equalized-odds difference across `sex`.
- Keep model comparisons in a clear table and preserve the distinction between
  predictive performance and fairness metrics.

## Scope

Do not add robustness or classification-threshold analysis. The final report
must be under 15 pages and should explain the fairness/performance trade-offs,
the limitations of the Adult dataset, its binary treatment of sex, missing
values, intersectional groups, and the limitations of statistical parity and
equalized odds.

## Coding guidelines

- Prefer small, reusable functions for loading, preprocessing, training,
  evaluation, and plotting.
- Keep sensitive attributes available for evaluation even when excluded from
  model features.
- Avoid data leakage: fit preprocessing only on training data.
- Make metric definitions and group labels explicit.
- Do not silently drop rows, features, or failed experiments; report such
  decisions clearly.
- Follow existing project conventions and update directly related
  documentation when behavior or commands change.
