# AQI Image Classifier API

Local FastAPI server wrapping the `AirQualityNet` checkpoint (`best_model.pth`
from the training notebook) as an HTTP endpoint.

## Setup

```bash
pip install -r requirements.txt
```

## Run

Put your checkpoint file somewhere accessible, then:

```bash
MODEL_PATH=/path/to/best_model.pth uvicorn app.main:app --reload --port 8000
```

If you don't set `MODEL_PATH`, it defaults to looking for `model.pth` in the
current directory.

## Endpoints

### `GET /health`
Quick check that the server is up and the model loaded successfully.

```bash
curl http://localhost:8000/health
```

### `POST /predict`
Send an image file (JPEG or PNG), get back the predicted AQI category plus
per-class probabilities.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/photo.jpg"
```

Response:
```json
{
  "predicted_class_index": 0,
  "predicted_label": "Good",
  "probabilities": {
    "Good": 0.57,
    "Moderate": 0.43,
    "Unhealthy": 0.0
  }
}
```

## Notes / things to double check

- **Class label mapping** (`CLASS_LABELS` in `app/main.py`) is inferred from
  the `AQIcategory()` function in the training notebook (0=Good, 1=Moderate,
  2=Unhealthy). This isn't stored in the checkpoint itself, so if label order
  ever changes on the training side, update it here too.
- **Preprocessing** in `inference_transform` must stay identical to
  `test_transform` from the notebook (224x224 resize, ImageNet normalization,
  no random flip). If the training pipeline changes, this needs to change too.
- This model was explicitly described as not very accurate (it's a learning
  project) — don't be surprised by confident-looking wrong predictions.
- Currently CPU/GPU auto-detected via `torch.cuda.is_available()`. Fine for
  local use; if you containerize this for cloud deployment later, you'll
  likely want a CPU-only Docker image unless you're paying for GPU instances.


# static file for testing

```
cd static && python3 -m http.server 5173
```