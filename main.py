from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
import pickle
import os
# -------- MODEL LOADING --------
MODEL_PATH = "model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    model = None
    print("Model loading failed:", e)

app = FastAPI()

# -------- CONFIG --------
API_KEY = "my-secret-api-key"  # you can change this later


# -------- INPUT SCHEMA --------
class VoiceInput(BaseModel):
    audio_base64: str
    language: str


# -------- HEALTH CHECK --------
@app.get("/")
def home():
    return {"message": "AI Voice Detection API is running"}


# -------- MAIN ENDPOINT --------
@app.post("/detect-voice")
def detect_voice(
    data: VoiceInput,
    x_api_key: Optional[str] = Header(None)
):
    # Check API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # Temporary fake logic (model will be added later)
    if len(data.audio_base64) < 20:
        raise HTTPException(status_code=400, detail="Audio data too short")

    return {
        "classification": "AI-generated",
        "confidence_score": 0.82,
        "explanation": [
            "Unnaturally stable pitch detected",
            "Low energy variation",
            "Absence of natural speech pauses"
        ]
    }



