# AI-Generated Voice Detection API  
*(AI for Fraud Detection & User Safety)*

## 📌 Project Overview

This project provides a **public REST API** that detects whether a given voice sample is **AI-generated or human-spoken** using a trained **machine learning classifier**.

The system supports **multi-language audio inputs** and returns a **classification result with a confidence score**, making it suitable for fraud detection, voice authentication, and user safety applications.

As per the problem statement, **no frontend/UI is included or required**. The solution is evaluated purely as an API.

---

## 🎯 Problem Statement

**AI-Generated Voice Detection (Multi-Language)**

Given an audio sample (Base64-encoded MP3) in one of the supported languages, determine whether the voice is:
- **Human**
- **AI-Generated**

The system returns:
- Classification result
- Confidence score
- Brief explanation

---

## 🌐 Supported Languages

The API supports the following languages (as required):

- English  
- Hindi  
- Malayalam  
- Tamil  
- Telugu  

---

## 🧠 Model & Approach

- Audio input is converted into **numeric audio features** (e.g., pitch, MFCC statistics, spectral features).
- These features are passed to a **trained scikit-learn classifier**.
- The model uses:
  - `predict()` for classification
  - `predict_proba()` to compute a confidence score
- Outputs are **not hard-coded**.
- Confidence scores are dynamically derived from model probabilities.

The trained model is serialized using **joblib** and loaded at API startup.

---

## 🔌 API Specification

### Base URL

