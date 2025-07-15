"""
Orchestrator Service
--------------------
* /predict   -> POST to MLflow-served classifier
* /generate  -> POST to Ollama LLM
* /health    -> liveness probe
* /metrics   -> Prometheus counter
"""

import os, httpx
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest

# ------------------------------------------------------------------ #
# 1. Configuration comes from env variables injected by docker-compose
# ------------------------------------------------------------------ #
CLS_EP = os.getenv("CLS_ENDPOINT")          # e.g. http://classifier:5001/invocations
LLM_EP = os.getenv("LLM_ENDPOINT")          # e.g. http://llm_service:11434/...
assert CLS_EP and LLM_EP, "Endpoints must be set via environment!"

# ------------------------------------------------------------------ #
# 2. FastAPI application with auto-doc enabled
# ------------------------------------------------------------------ #
app = FastAPI(
    title="Micro Orchestrator",
    version="1.0",
    docs_url="/docs", redoc_url="/redoc",
)

class Msg(BaseModel):
    text: str

# Prometheus metric – total requests to /predict
PRED_COUNTER = Counter("pred_requests_total",
                       "Number of /predict requests served")

# ------------------------------------------------------------------ #
# 3. End-points
# ------------------------------------------------------------------ #
@app.get("/health", tags=["system"])
def health():
    return {"status": "ok"}

@app.post("/predict", tags=["inference"])
async def predict(msg: Msg):
    PRED_COUNTER.inc()
    async with httpx.AsyncClient() as client:
        r = await client.post(CLS_EP, json={"inputs": [msg.text]})
    return r.json()

@app.post("/generate", tags=["inference"])
async def generate(msg: Msg):
    payload = {"messages": [{"role": "user", "content": msg.text}]}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(LLM_EP, json=payload)
    return r.json()

@app.get("/metrics", tags=["system"])
def metrics():
    """Prometheus-compatible metrics endpoint."""
    return generate_latest(), 200, {"Content-Type": "text/plain; version=0.0.4"}
