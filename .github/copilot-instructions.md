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

Described in `PLAN.MD`.

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
