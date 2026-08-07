"""Data drift detection between a reference dataset and an incoming batch, using Evidently."""

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

NUMERIC = ["age", "los_days", "n_diagnoses", "admitted_via_ed"]
CATEGORICAL = ["gender", "admission_type", "insurance", "marital_status", "race"]

DRIFT_SHARE_THRESHOLD = 0.3


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    data_def = DataDefinition(numerical_columns=NUMERIC, categorical_columns=CATEGORICAL)
    ref_ds = Dataset.from_pandas(reference, data_definition=data_def)
    cur_ds = Dataset.from_pandas(current, data_definition=data_def)

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=ref_ds, current_data=cur_ds).dict()

    drift_share = 0.0
    column_drift = {}
    for metric in result["metrics"]:
        if metric["metric_name"].startswith("DriftedColumnsCount"):
            drift_share = metric["value"]["share"]
        elif metric["metric_name"].startswith("ValueDrift"):
            column = metric["config"]["column"]
            column_drift[column] = metric["value"]

    return {
        "drift_share": drift_share,
        "drift_detected": drift_share >= DRIFT_SHARE_THRESHOLD,
        "column_drift": column_drift,
    }


if __name__ == "__main__":
    from pathlib import Path

    from data import build_features

    stream_dir = Path(__file__).resolve().parent.parent / "data" / "stream"
    reference = build_features()

    for batch_path in sorted(stream_dir.glob("batch_*.csv")):
        current = pd.read_csv(batch_path)
        result = detect_drift(reference, current)
        flag = "DRIFT" if result["drift_detected"] else "ok"
        print(f"{batch_path.name}: drift_share={result['drift_share']:.2f} [{flag}]")
