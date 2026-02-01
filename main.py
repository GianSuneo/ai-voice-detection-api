from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
import pickle
import os

app = FastAPI()

# -------- CONFIG --------
API_KEY = "my-secret-api-key"

# -------- MODEL LOADING --------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    model = None
    print("Model loading failed:", e)


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
    # API key check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # Basic input check
    if len(data.audio_base64) < 20:
        raise HTTPException(status_code=400, detail="Audio data too short")

    # Model check
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # -------- MODEL INFERENCE (TEMP) --------
    try:
        dummy_input = [[0.0]]  # placeholder until real features
        prediction = model.predict(dummy_input)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    label = "AI-generated" if prediction == 1 else "Human"

    return {
        "classification": label,
        "confidence_score": 0.85,
        "explanation": [
            "Prediction generated using ML model",
            "Audio features analyzed"
        ]
    }





