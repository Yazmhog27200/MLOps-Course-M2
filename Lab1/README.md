# Lab1 — MLflow Tracking (Solution/Template)

## Objectif
Instrumenter un entraînement scikit-learn avec **MLflow Tracking**, paramétrer des hyperparamètres, logger des métriques, des artefacts et enregistrer le modèle.

## Ce que vous allez modifier
- `train.py` : ajustez le modèle, les hyperparamètres et ce que vous logguez.
- `params.yaml` : changez les hyperparamètres et relancez l’entraînement.
- (Optionnel) Activez le backend de tracking distant (MLFLOW_TRACKING_URI).

## Utilisation rapide
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# lancer un serveur MLflow local (interface web)
mlflow ui --port 5000

# exécuter un run
python train.py --config params.yaml

# visualiser les runs: http://127.0.0.1:5000
```
