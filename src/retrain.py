"""Drift-triggered retraining and champion/challenger promotion logic.

Runnable standalone (for local testing) and also called from the Airflow DAG
in dags/retrain_pipeline.py — the orchestration layer just wires these
functions into tasks, all the actual logic lives here so it's testable
without a running Airflow instance.
"""

import sys
from pathlib import Path

import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "monitoring"))

from data import TARGET, build_features, split_train_holdout  # noqa: E402
from drift_detector import detect_drift  # noqa: E402
from train import CATEGORICAL, MLFLOW_TRACKING_URI, MODEL_NAME, NUMERIC, train  # noqa: E402

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
STREAM_DIR = Path(__file__).resolve().parent.parent / "data" / "stream"


def check_drift(batch_name: str) -> dict:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    train_pool, _holdout = split_train_holdout(build_features())
    current = pd.read_csv(STREAM_DIR / batch_name)
    return detect_drift(train_pool, current)


def retrain_with_batch(batch_name: str) -> dict:
    """Retrain on the training pool plus the drifted batch (with its true outcomes).

    The holdout split is never touched here, so it stays a clean evaluation set.
    """
    train_pool, _holdout = split_train_holdout(build_features())
    batch = pd.read_csv(STREAM_DIR / batch_name)
    combined = pd.concat([train_pool, batch[train_pool.columns]], ignore_index=True)
    _, metrics = train(combined)
    return metrics


def _client() -> MlflowClient:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def _holdout_auc(run_id: str) -> float:
    """Score a logged model's run against the fixed, never-trained-on holdout."""
    import mlflow.sklearn

    _train_pool, holdout = split_train_holdout(build_features())
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    X, y = holdout[CATEGORICAL + NUMERIC], holdout[TARGET]
    proba = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, proba)


def promote_if_better(challenger_run_id: str) -> dict:
    """Evaluate a newly retrained model against the current champion.

    If there's no champion yet, the new model becomes champion outright
    (nothing to canary against). Otherwise, if it beats the champion on the
    holdout, it's staged as the *challenger* — it starts serving a minority of
    live traffic (see api/serve.py) rather than immediately replacing the
    champion. finalize_canary() promotes it fully once its live performance
    has been reviewed.
    """
    client = _client()
    challenger_auc = _holdout_auc(challenger_run_id)

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    challenger_version = next(v for v in versions if v.run_id == challenger_run_id)

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        champion_auc = _holdout_auc(champion.run_id)
    except Exception:
        champion, champion_auc = None, -1.0

    better = challenger_auc >= champion_auc

    if champion is None:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, challenger_version.version)
        stage = "champion (no prior champion — bootstrapped directly)"
    elif better:
        client.set_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS, challenger_version.version)
        stage = "challenger (staged for canary rollout)"
    else:
        stage = "rejected (did not beat current champion)"

    return {
        "challenger_version": challenger_version.version,
        "challenger_auc": challenger_auc,
        "champion_version": champion.version if champion else None,
        "champion_auc": champion_auc,
        "promoted": better,
        "stage": stage,
    }


def finalize_canary() -> dict:
    """Promote the current challenger to champion, ending the canary period."""
    client = _client()
    challenger = client.get_model_version_by_alias(MODEL_NAME, CHALLENGER_ALIAS)
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, challenger.version)
    client.delete_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS)
    return {"new_champion_version": challenger.version}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="batch_5.csv")
    args = parser.parse_args()

    drift_result = check_drift(args.batch)
    print(f"drift check on {args.batch}: {drift_result}")

    if not drift_result["drift_detected"]:
        print("No significant drift — skipping retrain.")
    else:
        retrain_metrics = retrain_with_batch(args.batch)
        print(f"retrained: {retrain_metrics}")
        promotion = promote_if_better(retrain_metrics["run_id"])
        print(f"promotion decision: {promotion}")
