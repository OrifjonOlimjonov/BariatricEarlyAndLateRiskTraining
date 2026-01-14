import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names*")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, log_loss,
    mean_absolute_error, r2_score
)

from data_generate import generate_synthetic_bariatric
from features import add_missingness, make_snapshot, make_early_features


# ===== Model bundles =====
LGBM_PATH = "bariatric_lgbm_calibrated_bundle.joblib"
XGB_PATH  = "bariatric_xgb_calibrated_bundle.joblib"

SEED_DATA = 42
SEED_MISS = 7
SEED_SPLIT = 123
N_PATIENTS = 10_000


# ===== Uzbek column/task names =====
UZ_COLS_CLASSIFICATION = {
    "model": "Model",
    "task": "Vazifa",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "brier": "Brier skori (kalibratsiya)",
    "logloss": "LogLoss",
    "pos_rate": "Pozitiv ulushi",
    "proba_mean": "O‘rtacha risk (proba)",
}

UZ_COLS_REGRESSION = {
    "model": "Model",
    "task": "Vazifa",
    "mae": "MAE (o‘rtacha absolyut xato)",
    "r2": "R²",
}

UZ_TASK_NAMES = {
    "readmission_30d": "30 kun ichida qayta yotqizish",
    "complication_90d": "90 kun ichida asorat",
    "any_def_12m": "12 oy ichida har qanday yetishmovchilik",
    "twl_12m": "12 oyda umumiy vazn yo‘qotish (TWL)",
}

UZ_MODEL_NAMES = {
    "LGBM_cal": "LightGBM (kalibratsiyalangan)",
    "XGB_cal": "XGBoost (kalibratsiyalangan)",
}


def build_dataset():
    long_df, y = generate_synthetic_bariatric(n_patients=N_PATIENTS, seed=SEED_DATA)
    long_df = add_missingness(long_df, seed=SEED_MISS)

    X0 = make_snapshot(long_df, "discharge")
    Xearly = make_early_features(long_df)
    X = X0.join(Xearly, how="left")

    data = X.join(y.set_index("patient_id"), how="inner").reset_index()

    train_ids, test_ids = train_test_split(
        data["patient_id"], test_size=0.15, random_state=SEED_SPLIT, shuffle=True
    )
    test = data[data.patient_id.isin(test_ids)].copy()
    return test


def eval_classifier(model_name, clf, X_test, y_test, task_name):
    proba = clf.predict_proba(X_test)[:, 1]
    proba = np.clip(proba, 1e-15, 1 - 1e-15)

    return {
        "model": model_name,
        "task": task_name,
        "auroc": float(roc_auc_score(y_test, proba)),
        "auprc": float(average_precision_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "logloss": float(log_loss(y_test, proba)),
        "pos_rate": float(np.mean(y_test)),
        "proba_mean": float(np.mean(proba)),
    }


def eval_regressor(model_name, reg, X_test, y_test):
    pred = reg.predict(X_test)
    return {
        "model": model_name,
        "task": "twl_12m",
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }


def main():
    test = build_dataset()

    bundles = [
        ("LGBM_cal", joblib.load(LGBM_PATH)),
        ("XGB_cal",  joblib.load(XGB_PATH)),
    ]

    clf_tasks = ["readmission_30d", "complication_90d", "any_def_12m"]
    clf_rows = []
    reg_rows = []

    for model_name, bundle in bundles:
        feature_cols = bundle["feature_cols"]
        X_test = test[feature_cols].copy().astype("object")

        for t in clf_tasks:
            y_test = test[t].values
            clf = bundle["classifiers"][t]
            clf_rows.append(eval_classifier(model_name, clf, X_test, y_test, t))

        y_reg = test["twl_12m"].values
        reg = bundle["regressor"]
        reg_rows.append(eval_regressor(model_name, reg, X_test, y_reg))

    clf_df = pd.DataFrame(clf_rows).sort_values(["task", "auprc"], ascending=[True, False])
    reg_df = pd.DataFrame(reg_rows).sort_values(["task", "model"])

    # ---- Uzbek labels ----
    clf_df["task"] = clf_df["task"].map(UZ_TASK_NAMES).fillna(clf_df["task"])
    reg_df["task"] = reg_df["task"].map(UZ_TASK_NAMES).fillna(reg_df["task"])

    clf_df["model"] = clf_df["model"].map(UZ_MODEL_NAMES).fillna(clf_df["model"])
    reg_df["model"] = reg_df["model"].map(UZ_MODEL_NAMES).fillna(reg_df["model"])

    clf_df_uz = clf_df.rename(columns=UZ_COLS_CLASSIFICATION)
    reg_df_uz = reg_df.rename(columns=UZ_COLS_REGRESSION)

    print("=== Klassifikatsiya taqqoslash (Test) ===")
    print(clf_df_uz.to_string(index=False))

    print("\n=== Regressiya taqqoslash (Test) ===")
    print(reg_df_uz.to_string(index=False))

    # Save Uzbek CSV
    clf_df_uz.to_csv("model_compare_classification_uz.csv", index=False)
    reg_df_uz.to_csv("model_compare_regression_uz.csv", index=False)
    print("\nSaved: model_compare_classification_uz.csv, model_compare_regression_uz.csv")


if __name__ == "__main__":
    main()