from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Sentiment Analysis (Hugging Face)")

class InferenceRequest(BaseModel):
    text: str

class InferenceResponse(BaseModel):
    label: str
    score: float

# Load once at startup (default model is small & CPU friendly)
nlp = pipeline("sentiment-analysis")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    result = nlp(req.text)[0]
    # normalize label names (HF: POSITIVE/NEGATIVE)
    return InferenceResponse(label=result["label"].lower(), score=float(result["score"]))
