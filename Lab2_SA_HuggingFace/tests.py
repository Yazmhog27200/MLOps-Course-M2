import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_happy():
    r = client.post("/predict", json={"text": "I absolutely love it!"})
    assert r.status_code == 200
    body = r.json()
    assert "label" in body and "score" in body
    assert body["label"] in {"positive", "negative"}

def test_predict_empty():
    r = client.post("/predict", json={"text": "   "})
    assert r.status_code == 400
