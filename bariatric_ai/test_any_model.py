import warnings

from bariatric_ai.followup_optimization import optimize_followup_plan

warnings.filterwarnings("ignore", message="X does not have valid feature names*")

import numpy as np
import pandas as pd
import joblib

from recommendations import generate_recommendations

# 1) Choose model bundle here:
#MODEL_PATH = "bariatric_lgbm_bundle.joblib"
#MODEL_PATH = "bariatric_xgb_bundle.joblib"
#MODEL_PATH = "bariatric_xgb_calibrated_bundle.joblib"
MODEL_PATH = "bariatric_lgbm_calibrated_bundle.joblib"


bundle = joblib.load(MODEL_PATH)
feature_cols = bundle["feature_cols"]

# 2) Example patient (same as before)
x = {
    "age": 42, "sex": "F", "height_cm": 165,
    "preop_bmi": 46, "preop_weight_kg": 125,
    "smoking": 0, "dm2": 1, "htn": 1, "osa": 0, "gerd": 1, "nafld": 1, "depression": 0,
    "asa": 3, "op_duration_min": 105, "los_days": 3,

    # discharge features
    "diet_adherence": 0.70,
    "supplement_adherence": 0.65,
    "vomiting": 0,
    "reflux": 1,
    "diarrhea": 0,
    "dumping": 0,
    "protein_g_day": 60,
    "fluid_l_day": 1.6,
    "weight_kg": 123,
    "bmi": 45.2,
    "hba1c": 7.2,
    "ferritin": 70,
    "b12": 420,
    "vitd": 22,
    "albumin": 4.0,

    # early (2w+1m) aggregated
    "early_diet_adherence": 0.55,
    "early_supplement_adherence": 0.50,
    "early_vomiting": 1.0,
    "early_reflux": 1.0,
    "early_diarrhea": 0.0,
    "early_dumping": 0.0,
    "early_protein_g_day": 45,
    "early_fluid_l_day": 0.9,
    "early_weight_kg": 118,
    "early_bmi": 43.3,
    "early_hba1c": 6.8,
    "early_ferritin": 35,
    "early_b12": 240,
    "early_vitd": 18,
    "early_albumin": 3.5,
    "early_weight_change_kg": -5.0,
}

# 3) Build X with exact schema
X = pd.DataFrame([{c: None for c in feature_cols}], dtype=object)
for k, v in x.items():
    if k in X.columns:
        X.at[0, k] = v

X = X[feature_cols]

# 4) Predict
p_readm = float(bundle["classifiers"]["readmission_30d"].predict_proba(X)[:, 1][0])
p_comp  = float(bundle["classifiers"]["complication_90d"].predict_proba(X)[:, 1][0])
p_def   = float(bundle["classifiers"]["any_def_12m"].predict_proba(X)[:, 1][0])
twl     = float(bundle["regressor"].predict(X)[0])

print(f"=== Using model bundle: {MODEL_PATH} ===")
print("=== Predictions ===")
print("Risk readmission_30d :", f"{p_readm:.4f}")
print("Risk complication_90d:", f"{p_comp:.6f}")
print("Risk any_def_12m     :", f"{p_def:.4f}")
print("Pred TWL_12m         :", f"{twl:.4f}")

# 5) Recommendations (works the same for LGBM and XGB)
risks = {
    "p_readmission_30d": p_readm,
    "p_complication_90d": p_comp,
    "p_any_def_12m": p_def,
    "pred_twl_12m": twl,
}

plan = optimize_followup_plan(risks=risks, features=x)

print("\n=== Optimallashtirilgan follow-up reja (12 hafta) ===")
for item in plan["plan"]:
    print(f"Week {item['week']:>2}: {item['contact']}")

rec = generate_recommendations(risks=risks, features=x, mode="both")

print("\n=== Tavsiyalar (Bemor uchun) ===")
for i, item in enumerate(rec["items"], 1):
    print(f"{i}. [P{item['priority']}] {item['title']}")
    print("   -", item["patient_text"])

print("\n=== Tavsiyalar (Shifokor uchun) ===")
for i, item in enumerate(rec["items"], 1):
    print(f"{i}. [P{item['priority']}] {item['title']}")
    print("   -", item["clinician_text"])
    if item.get("triggers"):
        print("   - Trigger:", "; ".join(item["triggers"]))

print("\n" + rec["disclaimer_uz"])