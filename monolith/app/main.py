import pathlib
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------
# 1) Load the model ONCE during process start-up.
#    This is synchronous (blocking) but happens before the
#    server starts accepting requests, so no user sees delay.
# ---------------------------------------------------------
MODEL_DIR = pathlib.Path(__file__).parent.parent / "model"
model = mlflow.pyfunc.load_model(str(MODEL_DIR))

# ---------------------------------------------------------
# 2) Define the web API
# ---------------------------------------------------------
app = FastAPI(
    title="Message Classifier API",
    version="1.0.0",
    description="FastAPI service with baked-in MLflow model",
)

class Msg(BaseModel):
    text: str

@app.get("/health", tags=["system"])
def health():
    return {"status": "ok"}

@app.post("/predict", tags=["inference"])
def predict(msg: Msg):
    proba = model.predict([msg.text])[0]          # returns numpy.float32
    return {"probability": float(proba)}          # cast to JSON-serialisable
