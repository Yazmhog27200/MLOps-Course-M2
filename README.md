# Churn Classifier with Pipelines + MLflow

This project builds a **tabular classification pipeline** (scikit-learn `Pipeline` + `ColumnTransformer`)
and tracks everything with **MLflow**.

## Features
- Reproducible pipelines with preprocessing
- Cross-validation & tuning
- MLflow tracking & registry
- CLI + config-driven workflow
- Unit tests
- Makefile for automation

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# train
make train

# evaluate
make evaluate

# run tests
make test
```
