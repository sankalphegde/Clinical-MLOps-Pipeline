"""FastAPI scoring endpoint with champion/challenger canary routing.

If a challenger model is staged (see src/retrain.py::promote_if_better), a
configurable fraction of live traffic is routed to it instead of the champion,
and every prediction is logged with which model served it — that log is what
the monitoring dashboard reads to compare live challenger vs. champion
performance before finalize_canary() fully promotes it.
"""

import csv
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from train import CATEGORICAL, MLFLOW_TRACKING_URI, MODEL_NAME, NUMERIC  # noqa: E402

CANARY_TRAFFIC_SHARE = 0.2
# Overridable so the api and dashboard containers can share one log via a mounted volume.
PREDICTIONS_LOG = Path(
    os.environ.get(
        "PREDICTIONS_LOG_PATH",
        Path(__file__).resolve().parent.parent / "monitoring" / "predictions_log.csv",
    )
)

app = FastAPI(title="Clinical Mortality Risk Scoring")


class PatientFeatures(BaseModel):
    age: float
    gender: str
    admission_type: str
    insurance: str
    marital_status: str
    race: str
    los_days: float
    n_diagnoses: float
    admitted_via_ed: int


class PredictionResponse(BaseModel):
    mortality_risk: float
    model_used: Literal["champion", "challenger"]
    model_version: str


_models: dict[str, tuple] = {}


def _load_models():
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    champion_v = client.get_model_version_by_alias(MODEL_NAME, "champion")
    _models["champion"] = (
        mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion"),
        champion_v.version,
    )

    try:
        challenger_v = client.get_model_version_by_alias(MODEL_NAME, "challenger")
        _models["challenger"] = (
            mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@challenger"),
            challenger_v.version,
        )
    except Exception:
        _models.pop("challenger", None)


@app.on_event("startup")
def startup():
    _load_models()


def _log_prediction(features: PatientFeatures, model_used: str, version: str, risk: float):
    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    is_new = not PREDICTIONS_LOG.exists()
    with open(PREDICTIONS_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                ["timestamp", "model_used", "model_version", "mortality_risk", *CATEGORICAL, *NUMERIC]
            )
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                model_used,
                version,
                risk,
                *[getattr(features, c) for c in CATEGORICAL],
                *[getattr(features, n) for n in NUMERIC],
            ]
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "champion_version": _models["champion"][1] if "champion" in _models else None,
        "challenger_version": _models["challenger"][1] if "challenger" in _models else None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    use_challenger = "challenger" in _models and random.random() < CANARY_TRAFFIC_SHARE
    model_used = "challenger" if use_challenger else "champion"
    model, version = _models[model_used]

    import pandas as pd

    X = pd.DataFrame([features.model_dump()])[CATEGORICAL + NUMERIC]
    risk = float(model.predict_proba(X)[0, 1])

    _log_prediction(features, model_used, version, risk)

    return PredictionResponse(mortality_risk=risk, model_used=model_used, model_version=str(version))
