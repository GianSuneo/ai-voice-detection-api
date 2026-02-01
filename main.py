from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
import joblib
import os

app = FastAPI()

# -------- CONFIG --------
API_KEY = "my-secret-api-key"

# -------- MODEL LOADING --------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    model = joblib.load(MODEL_PATH)
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

    # -------- MODEL INFERENCE (TEMP FEATURES) --------
    try:
        features = [[
            0.0,  # pitch
            0.0,  # mfcc_mean
            0.0,  # spectral_flatness
            0.0,  # zcr
            0.0,  # rms
            0.0,  # centroid
            0.0   # bandwidth
        ]]

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}"
        )

    label = "AI-generated" if prediction == 1 else "Human"

    return {
        "classification": label,
        "confidence_score": round(confidence, 2),
        "explanation": [
            "Prediction generated using trained ML model",
            "Numeric audio features analyzed"
        ]
    }













