"""Airflow DAG orchestrating drift-triggered retraining.

check_drift -> (branch) -> retrain -> promote_if_better

All actual logic lives in src/retrain.py, which is fully testable standalone
without Airflow (`python retrain.py --batch batch_5.csv`). This DAG just wires
those functions into scheduled tasks.

Requires apache-airflow (see .venv-airflow) since Airflow needs Python <=3.12;
the rest of the project runs on the main .venv.
"""

import sys
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

DEFAULT_ARGS = {"owner": "sankalp", "retries": 1}


@dag(
    dag_id="clinical_mortality_retrain",
    schedule=None,  # triggered manually / by a monitoring job in this demo
    catchup=False,
    default_args=DEFAULT_ARGS,
    params={"batch_name": "batch_5.csv"},
)
def retrain_pipeline():
    @task
    def check_drift_task(**context) -> dict:
        from retrain import check_drift

        batch_name = context["params"]["batch_name"]
        result = check_drift(batch_name)
        result["batch_name"] = batch_name
        return result

    @task
    def retrain_task(drift_result: dict) -> dict:
        from retrain import retrain_with_batch

        if not drift_result["drift_detected"]:
            raise AirflowSkipException(
                f"drift_share={drift_result['drift_share']:.2f} below threshold — skipping retrain"
            )
        return retrain_with_batch(drift_result["batch_name"])

    @task
    def promote_task(retrain_result: dict) -> dict:
        from retrain import promote_if_better

        return promote_if_better(retrain_result["run_id"])

    drift_result = check_drift_task()
    retrain_result = retrain_task(drift_result)
    promote_task(retrain_result)


retrain_pipeline()
