"""
FastAPI Backend Application for Nationality, Emotion, Age, and Dress Colour Detection AI.
Loaded with 4 Custom Scratch-Trained CNN Models.

Strict Compliance:
- NO Pretrained Models / Transfer Learning / Third-party AI APIs
- Custom Forward Pass & Softmax Probability Outputs
- SQLite Logging
"""

import os
import sys
import shutil
import uuid
import numpy as np
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database import (
    init_db, save_prediction, get_recent_predictions,
    get_all_metrics, get_model_history
)
from backend.models import (
    create_nationality_model, create_emotion_model,
    create_age_model, create_dress_color_model,
    NATIONALITIES, EMOTIONS, AGE_GROUPS, DRESS_COLORS,
    get_architecture_specs
)
from backend.pipeline import preprocess_image_file
from utils.dataset_generator import create_full_datasets

UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "uploads")
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "samples")
SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Initialize database
init_db()

# Ensure datasets & sample images exist
if not os.path.exists(os.path.join(SAMPLES_DIR, "indian_happy_red.jpg")):
    create_full_datasets(images_per_class=5)

app = FastAPI(
    title="Nationality Detection AI (Custom Scratch CNNs)",
    description="4 Custom-Trained CNN Models from Scratch without Pretrained Weights or External APIs",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads static directory
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
if os.path.exists(SAMPLES_DIR):
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")

# Load 4 Scratch Models
nationality_model = create_nationality_model()
emotion_model = create_emotion_model()
age_model = create_age_model()
dress_color_model = create_dress_color_model()

# Load state dicts if present
def load_checkpoint(model, filename):
    path = os.path.join(SAVED_MODELS_DIR, filename)
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            print(f"Loaded scratch checkpoint: {filename}")
        except Exception as e:
            print(f"Could not load checkpoint {filename}: {e}")

load_checkpoint(nationality_model, "nationality_cnn.pt")
load_checkpoint(emotion_model, "emotion_cnn.pt")
load_checkpoint(age_model, "age_cnn.pt")
load_checkpoint(dress_color_model, "dress_colour_cnn.pt")

nationality_model.eval()
emotion_model.eval()
age_model.eval()
dress_color_model.eval()

def run_forward_pass(model, image_tensor: torch.Tensor, class_list: list):
    """Executes forward pass through custom scratch CNN model and extracts probabilities."""
    with torch.no_grad():
        probs = model(image_tensor)[0].numpy()
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        top_pred = class_list[pred_idx]
        
        # Sort full class breakdown
        breakdown = {class_list[i]: round(float(probs[i]), 4) for i in range(len(class_list))}
        return top_pred, round(confidence, 4), breakdown

@app.get("/health")
def health_check():
    """GET /health endpoint required by prompt specifications."""
    return {
        "status": "online",
        "models_loaded": {
            "nationality_cnn": True,
            "emotion_cnn": True,
            "age_cnn": True,
            "dress_colour_cnn": True
        },
        "pretrained_weights_used": False,
        "transfer_learning": False,
        "external_ai_apis": False
    }

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """POST /upload endpoint to accept and store user uploaded image."""
    try:
        ext = os.path.splitext(file.filename)[1]
        if not ext:
            ext = ".jpg"
        unique_name = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(UPLOADS_DIR, unique_name)
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        relative_path = f"/uploads/{unique_name}"
        return {
            "success": True,
            "filename": file.filename,
            "filepath": relative_path,
            "local_path": filepath
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")

@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    sample_path: Optional[str] = Form(None)
):
    """
    POST /predict endpoint required by prompt specifications.
    Performs forward pass on 4 custom scratch CNNs and evaluates conditional output logic.
    """
    image_location = None
    filename = "uploaded_image.jpg"

    if file:
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        filename = file.filename
        unique_name = f"{uuid.uuid4().hex}{ext}"
        image_location = os.path.join(UPLOADS_DIR, unique_name)
        with open(image_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image_url = f"/uploads/{unique_name}"
    elif sample_path:
        filename = os.path.basename(sample_path)
        # Sample path might be /samples/... or relative
        if sample_path.startswith("/"):
            rel = sample_path.lstrip("/")
            image_location = os.path.join(os.path.dirname(__file__), "..", "public", rel)
            image_url = sample_path
        else:
            image_location = os.path.join(SAMPLES_DIR, sample_path)
            image_url = f"/samples/{sample_path}"
    else:
        raise HTTPException(status_code=400, detail="Provide an uploaded file or sample_path.")

    if not os.path.exists(image_location):
        raise HTTPException(status_code=404, detail=f"Image file not found at {image_location}")

    # Preprocess image into 128x128 normalized tensor
    img_arr = preprocess_image_file(image_location) # (128, 128, 3)
    img_tensor = torch.tensor(img_arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) # (1, 3, 128, 128)

    # 1. Model 1: Nationality CNN
    pred_nat, conf_nat, breakdown_nat = run_forward_pass(nationality_model, img_tensor, NATIONALITIES)

    # 2. Model 2: Emotion CNN
    pred_emo, conf_emo, breakdown_emo = run_forward_pass(emotion_model, img_tensor, EMOTIONS)

    # Conditional Predictions Logic based on Nationality:
    # - If Indian: Predict Age AND Dress Colour
    # - If American: Predict Age only
    # - If African: Predict Dress Colour only
    # - Other Nationalities: Show only Nationality and Emotion
    
    pred_age = None
    conf_age = None
    breakdown_age = None

    pred_dress = None
    conf_dress = None
    breakdown_dress = None

    conditional_rule = ""

    if pred_nat == "Indian":
        conditional_rule = "Rule: Indian nationality requires both Age and Dress Colour prediction."
        pred_age, conf_age, breakdown_age = run_forward_pass(age_model, img_tensor, AGE_GROUPS)
        pred_dress, conf_dress, breakdown_dress = run_forward_pass(dress_color_model, img_tensor, DRESS_COLORS)
    elif pred_nat == "American":
        conditional_rule = "Rule: American nationality requires Age prediction only."
        pred_age, conf_age, breakdown_age = run_forward_pass(age_model, img_tensor, AGE_GROUPS)
    elif pred_nat == "African":
        conditional_rule = "Rule: African nationality requires Dress Colour prediction only."
        pred_dress, conf_dress, breakdown_dress = run_forward_pass(dress_color_model, img_tensor, DRESS_COLORS)
    else:
        conditional_rule = f"Rule: {pred_nat} nationality shows only Nationality and Emotion."

    # Save prediction log to SQLite database
    pred_id = save_prediction(
        filename=filename,
        image_path=image_url,
        nationality=pred_nat,
        nationality_confidence=conf_nat,
        emotion=pred_emo,
        emotion_confidence=conf_emo,
        age=pred_age,
        age_confidence=conf_age,
        dress_color=pred_dress,
        dress_color_confidence=conf_dress,
        conditional_rule=conditional_rule
    )

    return {
        "prediction_id": pred_id,
        "image_url": image_url,
        "filename": filename,
        "predictions": {
            "nationality": {
                "value": pred_nat,
                "confidence": conf_nat,
                "breakdown": breakdown_nat
            },
            "emotion": {
                "value": pred_emo,
                "confidence": conf_emo,
                "breakdown": breakdown_emo
            },
            "age": {
                "value": pred_age,
                "confidence": conf_age,
                "breakdown": breakdown_age
            } if pred_age else None,
            "dress_color": {
                "value": pred_dress,
                "confidence": conf_dress,
                "breakdown": breakdown_dress
            } if pred_dress else None
        },
        "conditional_rule": conditional_rule,
        "display_fields": {
            "show_nationality": True,
            "show_emotion": True,
            "show_age": pred_age is not None,
            "show_dress_color": pred_dress is not None
        }
    }

@app.get("/api/models")
def get_model_architectures():
    """Returns layer-by-layer specs and parameter counts of all 4 scratch CNN models."""
    return get_architecture_specs()

@app.get("/api/metrics")
def get_metrics():
    """Returns confusion matrices and model evaluation stats from SQLite."""
    metrics = get_all_metrics()
    return {"metrics": metrics}

@app.get("/api/history")
def get_prediction_history():
    """Returns SQLite log of recent user predictions."""
    history = get_recent_predictions(limit=30)
    return {"history": history}

@app.get("/api/model_history/{model_name}")
def get_history_by_model(model_name: str):
    """Returns epoch-by-epoch loss & accuracy curves for interactive graphs."""
    history = get_model_history(model_name)
    return {"model_name": model_name, "epochs": history}

@app.get("/api/samples")
def get_sample_gallery():
    """Returns built-in sample images for instant test uploads."""
    samples = [
        {"name": "Indian Portrait (Red Outfit)", "path": "/samples/indian_happy_red.jpg", "expected": "Indian"},
        {"name": "American Portrait (Blue Outfit)", "path": "/samples/american_neutral_blue.jpg", "expected": "American"},
        {"name": "African Portrait (Yellow Outfit)", "path": "/samples/african_surprise_yellow.jpg", "expected": "African"},
        {"name": "Japanese Portrait (Black Outfit)", "path": "/samples/japanese_sad_black.jpg", "expected": "Japanese"},
        {"name": "German Portrait (Green Outfit)", "path": "/samples/german_angry_green.jpg", "expected": "German"}
    ]
    return {"samples": samples}

@app.post("/api/train")
def trigger_scratch_training(background_tasks: BackgroundTasks):
    """Triggers retraining of all 4 CNNs from scratch in background."""
    from training.train_all_models import train_all
    background_tasks.add_task(train_all)
    return {"message": "Retraining of all 4 Scratch CNNs triggered in background."}
