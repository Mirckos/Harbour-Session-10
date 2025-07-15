"""
Classifier microservice
-----------------------
* Resolves <MODEL_NAME>@<MODEL_ALIAS> from the MLflow Registry
* Downloads and loads the model lazily at start-up
* Exposes:
    GET  /health
    POST /predict
"""
import os, mlflow.pyfunc
from mlflow.tracking import MlflowClient
from fastapi import FastAPI
from pydantic import BaseModel

# ---------- 0. Configuration ----------
TRACKING_URI  = os.environ["MLFLOW_URI"]          # e.g. http://mlflow:5000
MODEL_NAME    = os.getenv("MODEL_NAME", "msg_cls")
MODEL_ALIAS   = os.getenv("MODEL_ALIAS", "champion")

# ---------- 1. Load model once ----------
client = MlflowClient(tracking_uri=TRACKING_URI)          # official low-level API :contentReference[oaicite:3]{index=3}
info   = client.get_model_version_by_alias(MODEL_NAME,
                                           MODEL_ALIAS)   # resolves alias :contentReference[oaicite:4]{index=4}
model  = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{info.version}")  # loads artefact :contentReference[oaicite:5]{index=5}

# ---------- 2. FastAPI app ----------
app = FastAPI(title="Classifier", version="1.0")

class Msg(BaseModel):
    text: str

@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "model_version": info.version}

@app.post("/predict", tags=["inference"])
def predict(msg: Msg):
    proba = model.predict([msg.text])[0]
    return {"probability": float(proba), "model_version": info.version}
