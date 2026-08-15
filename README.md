# Bank Scoring MLOps

Bank credit scoring system with ClickHouse, FastAPI, CatBoost, Docker and reproducible ML pipeline.

## What it does

- pulls data from ClickHouse
- builds a training dataset from SQL extracts
- trains a baseline Logistic Regression model
- trains a CatBoost model
- compares models by AUC, Gini and KS
- serves scoring via FastAPI
- supports model reload without restart
- runs in Docker

## Project structure

- `app/` — FastAPI app
- `config/` — ClickHouse connection settings
- `ml/` — data loading, preprocessing, training, inference
- `scripts/` — runnable entry points
- `sql/` — SQL feature extracts
- `tests/` — smoke tests

## Main commands

Run the baseline model:
```powershell
python -m scripts.run_train