"""Train an in-hospital mortality risk model and log it to MLflow.

The MIMIC-IV demo is tiny (275 admissions, 15 deaths), so a single train/test
split would be noisy — cross-validation is used for the reported metrics, and
the final artifact is refit on all data before being logged/registered.
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data import TARGET, build_features, split_train_holdout

MLFLOW_TRACKING_URI = f"sqlite:///{Path(__file__).resolve().parent.parent / 'mlflow.db'}"
EXPERIMENT_NAME = "clinical-mortality-risk"
MODEL_NAME = "clinical-mortality-xgb"

CATEGORICAL = ["gender", "admission_type", "insurance", "marital_status", "race"]
NUMERIC = ["age", "los_days", "n_diagnoses", "admitted_via_ed"]


def build_pipeline(scale_pos_weight: float):
    from xgboost import XGBClassifier

    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train(df: pd.DataFrame | None = None):
    """Train on `df` if given, else on the training pool (holdout excluded)."""
    if df is None:
        df, _holdout = split_train_holdout(build_features())
    X, y = df[CATEGORICAL + NUMERIC], df[TARGET]

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    pipeline = build_pipeline(scale_pos_weight)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

    roc_auc = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)

    pipeline.fit(X, y)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "scale_pos_weight": scale_pos_weight}
        )
        mlflow.log_metrics({"cv_roc_auc": roc_auc, "cv_pr_auc": pr_auc, "n_samples": len(df)})
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
        )
        run_id = run.info.run_id

    print(f"CV ROC-AUC: {roc_auc:.3f} | CV PR-AUC: {pr_auc:.3f} | run_id: {run_id}")
    return pipeline, {"roc_auc": roc_auc, "pr_auc": pr_auc, "run_id": run_id}


if __name__ == "__main__":
    train()
