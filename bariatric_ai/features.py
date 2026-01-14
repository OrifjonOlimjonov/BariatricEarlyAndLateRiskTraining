# features.py
import numpy as np
import pandas as pd

def add_missingness(long_df, seed=7):
    """Simulate missing lab values like real EHR."""
    rng = np.random.default_rng(seed)
    df = long_df.copy()

    # higher missingness early
    miss_map = {
        "discharge": 0.15,
        "2w": 0.35,
        "1m": 0.30,
        "3m": 0.20,
        "6m": 0.15,
        "12m": 0.10
    }
    lab_cols = ["hba1c", "ferritin", "b12", "vitd", "albumin"]
    for tp, p in miss_map.items():
        idx = df["timepoint"] == tp
        for c in lab_cols:
            m = rng.random(idx.sum()) < p
            df.loc[idx, c] = df.loc[idx, c].mask(m)
    return df

def make_snapshot(long_df, snapshot="discharge"):
    """Return patient-level table using a single timepoint record."""
    df = long_df[long_df.timepoint == snapshot].copy()
    df = df.drop(columns=["timepoint", "twl_12m_true"], errors="ignore")
    # sex to categorical string, keep as is; later OneHotEncoder
    return df.set_index("patient_id")

def make_early_features(long_df):
    """Aggregate 2w and 1m into summary features for dynamic risk update."""
    df = long_df[long_df.timepoint.isin(["2w", "1m"])].copy()

    agg = df.groupby("patient_id").agg({
        "diet_adherence": "mean",
        "supplement_adherence": "mean",
        "vomiting": "mean",
        "reflux": "mean",
        "diarrhea": "mean",
        "dumping": "mean",
        "protein_g_day": "mean",
        "fluid_l_day": "mean",
        "weight_kg": "last",
        "bmi": "last",
        "hba1c": "mean",
        "ferritin": "mean",
        "b12": "mean",
        "vitd": "mean",
        "albumin": "mean",
    }).rename(columns=lambda c: f"early_{c}")

    # also weight change from discharge to 1m if present
    d0 = long_df[long_df.timepoint=="discharge"][["patient_id","weight_kg"]].set_index("patient_id")
    agg["early_weight_change_kg"] = agg["early_weight_kg"] - d0["weight_kg"]

    return agg