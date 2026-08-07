"""Load and feature-engineer the MIMIC-IV Demo dataset for in-hospital mortality prediction.

Uses the open-access MIMIC-IV Clinical Database Demo (physionet.org/content/mimic-iv-demo),
a ~100-patient subset that requires no credentialing. The full credentialed MIMIC-IV can be
swapped in via the same schema without changing any code below.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TARGET = "hospital_expire_flag"


def load_raw() -> dict[str, pd.DataFrame]:
    return {
        "admissions": pd.read_csv(DATA_DIR / "admissions.csv", parse_dates=["admittime", "dischtime"]),
        "patients": pd.read_csv(DATA_DIR / "patients.csv"),
        "diagnoses": pd.read_csv(DATA_DIR / "diagnoses_icd.csv"),
    }


def build_features(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    tables = tables or load_raw()
    adm, pat, dx = tables["admissions"], tables["patients"], tables["diagnoses"]

    df = adm.merge(pat, on="subject_id", how="left")

    df["los_days"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400

    dx_counts = dx.groupby("hadm_id").size().rename("n_diagnoses")
    df = df.merge(dx_counts, on="hadm_id", how="left")
    df["n_diagnoses"] = df["n_diagnoses"].fillna(0)

    came_via_ed = df["edregtime"].notna()
    df["admitted_via_ed"] = came_via_ed.astype(int)

    keep = [
        "hadm_id",
        "anchor_age",
        "gender",
        "admission_type",
        "insurance",
        "marital_status",
        "race",
        "los_days",
        "n_diagnoses",
        "admitted_via_ed",
        TARGET,
    ]
    df = df[keep].rename(columns={"anchor_age": "age"})

    categorical_cols = ["gender", "admission_type", "insurance", "marital_status", "race"]
    for col in categorical_cols:
        df[col] = df[col].fillna("unknown").astype("category")

    return df


def split_train_holdout(df: pd.DataFrame, holdout_frac: float = 0.2, seed: int = 42):
    """Fixed stratified holdout, carved out once and never trained on again.

    Used exclusively for champion/challenger evaluation during retraining, so
    the comparison isn't contaminated by data the challenger was trained on.
    """
    from sklearn.model_selection import train_test_split

    train_df, holdout_df = train_test_split(
        df, test_size=holdout_frac, stratify=df[TARGET], random_state=seed
    )
    return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True)


if __name__ == "__main__":
    features = build_features()
    print(features.shape)
    print(features[TARGET].value_counts())
    print(features.head())
