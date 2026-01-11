from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from pillow_heif import register_heif_opener
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import logging

app = FastAPI(title="Precision Farming ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👇 allow PIL to open HEIC/HEIF images (iPhone)
register_heif_opener()

MODEL_PATH = "model/tomato_disease_model.keras"
model = load_model(MODEL_PATH)

CLASS_NAMES = ["healthy", "leaf_blight", "rust", "pest_damage"]
CONFIDENCE_THRESHOLD = 0.5

def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    return x

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    size = len(contents or b"")
    logging.info(f"Received file name={file.filename} type={file.content_type} size={size}")

    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file")

    try:
        img = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        logging.exception("Error opening image")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    x = preprocess_image(img)
    preds = model.predict(x)[0]

    top_indices = preds.argsort()[-3:][::-1]
    top3 = [
        {"label": CLASS_NAMES[i], "confidence": float(preds[i])}
        for i in top_indices
    ]

    best_idx = int(np.argmax(preds))
    best_label = CLASS_NAMES[best_idx]
    best_conf = float(preds[best_idx])

    if best_conf < CONFIDENCE_THRESHOLD:
        return {"label": "uncertain", "confidence": best_conf, "top3": top3}

    return {"label": best_label, "confidence": best_conf, "top3": top3}
