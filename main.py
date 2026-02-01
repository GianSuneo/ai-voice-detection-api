from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
import joblib
import os
import base64
import io
import numpy as np
import librosa

# ---------------- APP INIT ----------------
app = FastAPI(title="AI-Generated Voice Detection API")

# ---------------- CONFIG ----------------
API_KEY = "my-secret-api-key"

# ---------------- MODEL LOADING ----------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    model = None
    print("❌ Model loading failed:", e)

# ---------------- INPUT SCHEMA ----------------
class VoiceInput(BaseModel):
    audio_base64: str
    language: str  # English, Hindi, Malayalam, Tamil, Telugu

# ---------------- HEALTH CHECK ----------------
@app.get("/")
def home():
    return {"message": "AI Voice Detection API is running"}

# ---------------- FEATURE EXTRACTION ----------------
def extract_audio_features(audio_base64: str):
    # Decode Base64 → bytes
    audio_bytes = base64.b64decode(audio_base64)
    audio_buffer = io.BytesIO(audio_bytes)

    # Load audio
    y, sr = librosa.load(audio_buffer, sr=None)

    # Feature extraction (matches training shape = 7)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs)

    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0

    return [[
        pitch,
        mfcc_mean,
        flatness,
        zcr,
        rms,
        centroid,
        bandwidth
    ]]

# ---------------- MAIN ENDPOINT ----------------
@app.post("/detect-voice")
def detect_voice(
    data: VoiceInput,
    x_api_key: Optional[str] = Header(None)
):
    # ---- API KEY VALIDATION ----
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # ---- INPUT VALIDATION ----
    if not data.audio_base64 or len(data.audio_base64) < 20:
        raise HTTPException(status_code=400, detail="Invalid or empty audio input")

    if data.language not in ["English", "Hindi", "Malayalam", "Tamil", "Telugu"]:
        raise HTTPException(status_code=400, detail="Unsupported language")

    # ---- MODEL CHECK ----
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # ---- MODEL INFERENCE ----
    try:
        features = extract_audio_features(data.audio_base64)

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # ---- OUTPUT FORMATTING ----
    label = "AI-generated" if prediction == 1 else "Human"
    confidence_score = round(float(max(probabilities)), 2)

    return {
        "classification": label,
        "confidence_score": confidence_score,
        "explanation": [
            "Audio converted to numeric features before inference",
            "Confidence score derived from model probability distribution"
        ]
    }
















