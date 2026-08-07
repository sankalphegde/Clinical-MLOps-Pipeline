# Clinical MLOps Pipeline

A closed-loop MLOps system for a clinical risk model: data drift monitoring, automated retraining, champion/challenger canary rollout, and live serving — not just a trained model in a notebook.

Built around in-hospital mortality prediction on the [MIMIC-IV Clinical Database Demo](https://physionet.org/content/mimic-iv-demo/2.2/) (open access, no credentialing required — the full credentialed MIMIC-IV can be swapped in via the same schema).

## Why this exists

Most portfolio ML projects stop at "trained a model, here's the accuracy." The harder, more realistic problem is: what happens after deployment, when the incoming data distribution shifts? This project answers that end to end:

**drift detection → automated retraining → holdout evaluation → canary staging → live traffic routing → monitoring**

## Architecture

| Component | Tool | What it does |
|---|---|---|
| Model | XGBoost | In-hospital mortality risk from admission/patient features |
| Experiment tracking + registry | MLflow | Versioning, champion/challenger aliases |
| Drift detection | Evidently | Statistical drift tests (K-S, Z-test) on incoming batches |
| Orchestration | Airflow | `check_drift → retrain → promote` DAG |
| Serving | FastAPI | Scoring endpoint with canary traffic routing |
| Monitoring | Streamlit | Drift over time, registry state, live canary performance |
| Deployment | Docker Compose | All of the above as one stack |

## The data

275 admissions from the MIMIC-IV demo, 15 deaths (5.5% positive rate). This is intentionally small and real — the point of this project is the pipeline around the model, not squeezing out state-of-the-art accuracy from a tiny dataset. Because of the small N:

- Metrics are cross-validated (5-fold), not a single noisy train/test split.
- A fixed 20% holdout is carved out once and **never trained on**, used exclusively for champion/challenger comparison — the first version of this project didn't do this and produced a meaningless 1.0 AUC from evaluating a model on data it had partly trained on. Fixed by making the holdout genuinely held out.

## Simulating production drift

MIMIC-IV demo is a static historical snapshot — there's no live feed to monitor. `simulate_stream.py` generates 6 synthetic batches by bootstrap-resampling the real feature distributions: batches 0–2 stay close to the reference distribution, batches 3–5 have a deliberate shift injected (older, sicker patients, more ED admissions). This is explicitly synthetic and documented as such.

Drift share across the batches (threshold for triggering retraining: 0.3):

| Batch | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Drift share | 0.11 | 0.00 | 0.11 | 0.22 | 0.22 | **0.44** |

Batch 5 crosses the threshold and triggers retraining.

## Champion/challenger promotion

When drift is detected, the pipeline retrains on the training pool plus the drifted batch, then evaluates the new model against the current champion **on the untouched holdout**:

- Champion (v1, trained on original data only): holdout AUC **0.590**
- Challenger (retrained with drifted batch): holdout AUC **0.654**

The challenger beats the champion, so it's staged as `challenger` — not immediately swapped in. The serving layer routes a minority of live traffic to it (20% by default); `finalize_canary()` promotes it fully once its live performance has been reviewed.

## Running it

**Locally:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src
python data.py                          # sanity check load
python train.py                         # bootstrap champion model
python simulate_stream.py               # generate synthetic production batches
python retrain.py --batch batch_5.csv   # drift check -> retrain -> promote
uvicorn ../api/serve:app --reload --app-dir ../api   # scoring endpoint
streamlit run ../monitoring/dashboard.py             # monitoring dashboard
```

**Full stack (Docker Compose):**
```bash
docker compose up -d mlflow
docker compose run --rm api python /app/src/train.py   # bootstrap champion
docker compose up -d api dashboard airflow
```
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Airflow: http://localhost:8082
- MLflow: http://localhost:5001

Trigger the DAG manually from the Airflow UI, or via CLI:
```bash
docker compose exec airflow airflow dags test clinical_mortality_retrain 2026-01-01 --conf '{"batch_name": "batch_5.csv"}'
```

## Project structure

```text
.
├── data/                    # MIMIC-IV demo tables + simulated production batches
├── src/
│   ├── data.py              # load, feature engineering, fixed holdout split
│   ├── train.py             # XGBoost + MLflow logging, champion bootstrap
│   ├── simulate_stream.py   # synthetic production traffic with injected drift
│   └── retrain.py           # drift check, retrain, champion/challenger promotion
├── monitoring/
│   ├── drift_detector.py    # Evidently-based drift detection
│   └── dashboard.py         # Streamlit monitoring dashboard
├── api/serve.py             # FastAPI scoring endpoint with canary routing
├── dags/retrain_pipeline.py # Airflow DAG wiring the retrain logic into tasks
├── docker/                  # Dockerfiles for mlflow, api, dashboard, airflow
├── docker-compose.yml
└── .github/workflows/ci.yml # runs the full pipeline + builds all images on push
```

## What's real vs. simulated

To be upfront about scope: the model, training, drift detection, retraining logic, promotion decisions, and serving are all real and verified working end to end (including inside Docker and Airflow). What's synthetic is the "incoming production traffic" — since MIMIC-IV demo is a static snapshot with no live feed, `simulate_stream.py` generates it by resampling the real data with a deliberate injected shift, clearly labeled as such rather than presented as real patient data.
