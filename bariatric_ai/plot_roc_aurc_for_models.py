import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, roc_auc_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

LONG_PATH = os.getenv("BARIATRIC_LONG_PATH", "data_out/bariatric_long_10000_seed42.csv")
OUT_PATH  = os.getenv("BARIATRIC_OUTCOMES_PATH", "data_out/bariatric_outcomes_10000_seed42.csv")

def build_features(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df["timepoint"] = df["timepoint"].astype(str).str.lower().str.strip()

    discharge = (
        df[df["timepoint"] == "discharge"]
        .sort_values(["patient_id"])
        .groupby("patient_id", as_index=False)
        .first()
    )

    early = df[df["timepoint"].isin(["2w", "1m"])].copy()
    exclude = {"patient_id", "timepoint"}
    num_cols = [c for c in early.columns if c not in exclude and pd.api.types.is_numeric_dtype(early[c])]

    early_agg = early.groupby("patient_id")[num_cols].agg(["mean", "min", "max"])
    early_agg.columns = [f"early_{c}_{stat}" for c, stat in early_agg.columns]
    early_agg = early_agg.reset_index()

    feat = discharge.drop(columns=["timepoint"], errors="ignore").merge(early_agg, on="patient_id", how="left")

    for c in feat.columns:
        if pd.api.types.is_numeric_dtype(feat[c]):
            feat[c] = feat[c].fillna(feat[c].median())
        else:
            feat[c] = feat[c].fillna("UNK")
    return feat

def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    num_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

def main(task="any_def_12m"):
    long_df = pd.read_csv(LONG_PATH)
    outcomes = pd.read_csv(OUT_PATH)

    X = build_features(long_df)
    data = X.merge(outcomes, on="patient_id", how="inner")

    outcome_cols = [c for c in outcomes.columns if c != "patient_id"]
    feature_cols = [c for c in data.columns if c not in (["patient_id"] + outcome_cols)]

    # Stratify: shu task bo‘yicha
    train_df, test_df = train_test_split(
        data, test_size=0.15, random_state=42, stratify=data[task]
    )

    X_train = train_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()
    y_train = train_df[task].astype(int).to_numpy()
    y_test  = test_df[task].astype(int).to_numpy()

    pre_tree = make_preprocessor(X_train, scale_numeric=False)
    pre_svm  = make_preprocessor(X_train, scale_numeric=True)

    models = {
        "LightGBM": (pre_tree, LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, random_state=42
        )),
        "XGBoost": (pre_tree, XGBClassifier(
            n_estimators=800, learning_rate=0.03, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=42, eval_metric="logloss"
        )),
        "SVM": (pre_svm, SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", random_state=42, probability=False
        )),
    }

    plt.figure(figsize=(7,5))

    # diagonal baseline
    plt.plot([0, 1], [0, 1], "r--", label="Random (AUC = 0.50)")

    for name, (pre, base_model) in models.items():
        pipe = Pipeline([("pre", pre), ("model", base_model)])
        cal = CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
        cal.fit(X_train, y_train)

        y_score = cal.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)

        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")

    plt.title(f"ROC–AUC Curves (Solishtirma) — {task}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"roc_auc_curves_{task}.png", dpi=200)
    plt.close()

    print(f"Saved: roc_auc_curves_{task}.png")

if __name__ == "__main__":
    main(task="any_def_12m")  # xohlasangiz: readmission_30d / complication_90d