import json
import logging
import os
import sys
from contextlib import asynccontextmanager

import joblib
import numpy as np
import sklearn
from fastapi import FastAPI
from pydantic import Field, create_model

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    stream=sys.stdout, level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("triage")

_BASES = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
          "concavity", "concave_points", "symmetry", "fractal_dimension"]
FEATURES = ([f"mean_{b}" for b in _BASES]
            + [f"{b}_error" for b in _BASES]
            + [f"worst_{b}" for b in _BASES])

PredictionRequest = create_model(
    "PredictionRequest",
    **{f: (float, Field(..., ge=0, allow_inf_nan=False)) for f in FEATURES},
)

state: dict = {}


def _threshold() -> float:
    try:
        return float(os.getenv("DECISION_THRESHOLD", "0.5"))
    except ValueError:
        log.error("DECISION_THRESHOLD must be a number")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    card_path = os.path.join(os.path.dirname(model_path), "model_card.json")
    if not os.path.exists(model_path) or not os.path.exists(card_path):
        log.error("Model artifact not found at %s (or model card at %s). Refusing to start.",
                  model_path, card_path)
        raise SystemExit(1)
    state["model"] = joblib.load(model_path)
    with open(card_path) as f:
        state["card"] = json.load(f)
    if state["card"]["feature_names"] != FEATURES:
        log.error("Model card feature names do not match the API contract")
        raise SystemExit(1)
    state["threshold"] = _threshold()
    state["version"] = os.getenv("MODEL_VERSION", state["card"]["model_version"])
    log.info("Model loaded: family=%s version=%s threshold=%s",
             state["card"]["model_family"], state["version"], state["threshold"])
    yield
    state.clear()


app = FastAPI(title="Breast Cancer Triage API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictionRequest):
    X = np.array([[getattr(req, f) for f in FEATURES]], dtype=float)
    model = state["model"]
    mal_idx = list(model.classes_).index(0)  # target 0 = malignant
    p_mal = float(model.predict_proba(X)[0][mal_idx])
    label = "malignant" if p_mal >= state["threshold"] else "benign"
    log.info("prediction label=%s p_malignant=%.4f", label, p_mal)
    return {
        "label": label,
        "malignant_probability": round(p_mal, 4),
        "threshold": state["threshold"],
        "model_version": state["version"],
    }


@app.get("/model/info")
def model_info():
    card = state["card"]
    return {
        "model_version": state["version"],
        "model_family": card["model_family"],
        "scikit_learn_version": sklearn.__version__,
        "git_commit": os.getenv("GIT_SHA", "unknown"),
        "decision_threshold": state["threshold"],
        "feature_count": len(card["feature_names"]),
        "metrics": card["metrics"]["test"],
    }


@app.get("/stats")
def stats():
    # Placeholder until Milestone 4 (Redis counter)
    return {"predictions_served": None, "note": "redis counter not enabled"}
