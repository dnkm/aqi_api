"""
FastAPI server exposing the AQI image classifier as an HTTP endpoint.

Run locally with:
    uvicorn app.main:app --reload --port 8000

Then POST an image to /predict, e.g.:
    curl -X POST http://localhost:8000/predict -F "file=@some_photo.jpg"
"""
import io
import os
from fastapi.middleware.cors import CORSMiddleware

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

from app.model import load_model


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
CHECKPOINT_PATH = os.environ.get("MODEL_PATH", "model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3

# Maps the model's output index -> the AQIcategory() buckets defined in the
# training notebook: 0 = Good (AQI 0-50), 1 = Moderate (51-100), 2 = Unhealthy (101+)
CLASS_LABELS = {
    0: "Good",
    1: "Moderate",
    2: "Unhealthy",
}

# Must exactly match `test_transform` from the training notebook, since the
# model only saw images preprocessed this way during training/validation.
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225]),
])

# -----------------------------------------------------------------------
# App + model loading (loaded once at startup, not per-request)
# -----------------------------------------------------------------------
app = FastAPI(title="AQI Image Classifier API")

# Comma-separated list, e.g. "http://localhost:5173,https://your-frontend.onrender.com"
_default_origins = "http://localhost:3000,http://localhost:5173"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

# Serve the bundled static frontend. Resolved relative to the project root so it
# works no matter which directory uvicorn is launched from.
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.on_event("startup")
def startup_load_model():
    global model
    if not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(
            f"Model checkpoint not found at '{CHECKPOINT_PATH}'. "
            f"Set the MODEL_PATH environment variable or place model.pth in the working directory."
        )
    model = load_model(CHECKPOINT_PATH, num_classes=NUM_CLASSES, device=DEVICE)
    print(f"Model loaded from {CHECKPOINT_PATH} on device={DEVICE}")


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file as an image.")

    input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probabilities).item())

    return JSONResponse({
        "predicted_class_index": pred_idx,
        "predicted_label": CLASS_LABELS.get(pred_idx, "Unknown"),
        "probabilities": {
            CLASS_LABELS.get(i, str(i)): round(float(p), 4)
            for i, p in enumerate(probabilities)
        },
        "modelVersion": "0.1.0",
    })
