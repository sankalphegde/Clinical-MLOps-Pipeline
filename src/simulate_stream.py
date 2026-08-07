"""Simulate incoming production traffic as a series of batches.

The MIMIC-IV demo (275 admissions) is a static historical snapshot, so there's
no real live feed to monitor. This generates synthetic batches by bootstrap-
resampling the real feature distributions — batches 0-2 stay close to the
reference distribution, batches 3+ have a deliberate population shift injected
(older, sicker patients admitted more often via the ED) so the drift-detection
and retraining loop has something real to react to. This is explicitly
synthetic and is documented as such — it is not claimed to be real patient data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data import build_features

STREAM_DIR = Path(__file__).resolve().parent.parent / "data" / "stream"
N_BATCHES = 6
BATCH_SIZE = 60
DRIFT_STARTS_AT_BATCH = 3


def _bootstrap_batch(reference: pd.DataFrame, rng: np.random.Generator, drift: bool) -> pd.DataFrame:
    idx = rng.integers(0, len(reference), size=BATCH_SIZE)
    batch = reference.iloc[idx].reset_index(drop=True).copy()

    if drift:
        # Simulate a population shift: older patients, more comorbidities,
        # more ED admissions — the kind of shift a hospital might see with a
        # change in referral patterns or a seasonal illness spike.
        batch["age"] = (batch["age"] + rng.normal(15, 5, size=len(batch))).clip(18, 100).round()
        batch["n_diagnoses"] = (batch["n_diagnoses"] * rng.uniform(1.4, 1.8, size=len(batch))).round()
        flip_to_ed = rng.random(len(batch)) < 0.3
        batch.loc[flip_to_ed, "admitted_via_ed"] = 1

    return batch


def generate_stream(seed: int = 7) -> list[Path]:
    STREAM_DIR.mkdir(parents=True, exist_ok=True)
    reference = build_features()
    rng = np.random.default_rng(seed)

    paths = []
    for batch_id in range(N_BATCHES):
        drift = batch_id >= DRIFT_STARTS_AT_BATCH
        batch = _bootstrap_batch(reference, rng, drift)
        path = STREAM_DIR / f"batch_{batch_id}.csv"
        batch.to_csv(path, index=False)
        paths.append(path)
        print(f"batch_{batch_id}: drift_injected={drift}, mean_age={batch['age'].mean():.1f}")

    return paths


if __name__ == "__main__":
    generate_stream()
