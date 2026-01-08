from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Precision Farming ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    img = Image.open(io.BytesIO(contents))

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
