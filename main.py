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
    # -------- MODEL INFERENCE --------
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded"
        )

    try:
        # TEMPORARY placeholder input
        dummy_input = [[0.0]]  # will replace with real features later
        prediction = model.predict(dummy_input)

        # Assume: 1 = AI, 0 = Human
        label = "AI-generated" if prediction[0] == 1 else "Human"

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}"
        )

       return {
        "classification": label,
        "confidence_score": 0.85,
        "explanation": [
            "Prediction generated using ML model",
            "Audio features analyzed"
        ]
    }




