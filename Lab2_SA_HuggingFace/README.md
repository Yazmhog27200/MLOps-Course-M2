# Lab2 — Sentiment Analysis (Hugging Face) Microservice

## Objectif
Déployer un microservice **FastAPI** qui fait de l'analyse de sentiments via `transformers` (pipeline). Inclut tests, Docker et CI (workflow GitHub Actions).

## Ce que vous allez modifier
- `app/main.py` : le modèle, la logique de pré/post-traitement, les schémas.
- `requirements.txt` : pinner/mettre à jour les libs.
- `tests/test_app.py` : cas de test supplémentaires.
- `Dockerfile` / `Makefile` : options d’exécution.
- `.github/workflows/ci.yaml` : pipeline (lint + tests).

## Lancer en local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# test
pytest -q
# requête
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"text":"I love this!"}'
```

## Docker
```bash
docker build -t sa-hf:latest .
docker run -p 8000:8000 sa-hf:latest
```
