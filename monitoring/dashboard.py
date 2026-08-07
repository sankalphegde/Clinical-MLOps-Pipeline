"""Streamlit dashboard: drift over time, registry state, and live canary performance."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import build_features, split_train_holdout  # noqa: E402
from drift_detector import detect_drift  # noqa: E402
from train import MLFLOW_TRACKING_URI, MODEL_NAME  # noqa: E402

STREAM_DIR = Path(__file__).resolve().parent.parent / "data" / "stream"
PREDICTIONS_LOG = Path(__file__).resolve().parent / "predictions_log.csv"

st.set_page_config(page_title="Clinical MLOps Monitoring", layout="wide")
st.title("Clinical Mortality Model — Monitoring")

st.header("Data drift across simulated production batches")
train_pool, _holdout = split_train_holdout(build_features())

drift_rows = []
for batch_path in sorted(STREAM_DIR.glob("batch_*.csv")):
    current = pd.read_csv(batch_path)
    result = detect_drift(train_pool, current)
    drift_rows.append({"batch": batch_path.stem, "drift_share": result["drift_share"]})

if drift_rows:
    drift_df = pd.DataFrame(drift_rows)
    st.line_chart(drift_df.set_index("batch")["drift_share"])
    st.caption("Threshold for triggering retraining: 0.3 drift share")
else:
    st.info("No simulated batches found — run `python src/simulate_stream.py` first.")

st.header("Model registry state")
try:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    col1, col2 = st.columns(2)
    with col1:
        try:
            champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
            st.metric("Champion version", champion.version)
        except Exception:
            st.metric("Champion version", "none")
    with col2:
        try:
            challenger = client.get_model_version_by_alias(MODEL_NAME, "challenger")
            st.metric("Challenger version (canary)", challenger.version)
        except Exception:
            st.metric("Challenger version (canary)", "none")
except Exception as e:
    st.warning(f"Could not reach MLflow: {e}")

st.header("Live canary traffic")
if PREDICTIONS_LOG.exists():
    log = pd.read_csv(PREDICTIONS_LOG)
    split = log["model_used"].value_counts(normalize=True) * 100
    st.write(f"Traffic split — champion: {split.get('champion', 0):.1f}% | challenger: {split.get('challenger', 0):.1f}%")

    by_model = log.groupby("model_used")["mortality_risk"].agg(["count", "mean"])
    st.dataframe(by_model.rename(columns={"count": "n_predictions", "mean": "avg_predicted_risk"}))
    st.line_chart(log.reset_index()["mortality_risk"])
else:
    st.info("No predictions logged yet — hit the /predict endpoint in api/serve.py to generate traffic.")
