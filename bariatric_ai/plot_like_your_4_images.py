import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix

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

def plot_confusion(cm, title, filename):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap="viridis")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", color="gold", fontsize=12)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def main(task="any_def_12m"):
    long_df = pd.read_csv(LONG_PATH)
    outcomes = pd.read_csv(OUT_PATH)

    X = build_features(long_df)
    data = X.merge(outcomes, on="patient_id", how="inner")

    # Outcomes columns must not be features
    outcome_cols = [c for c in outcomes.columns if c != "patient_id"]
    feature_cols = [c for c in data.columns if c not in (["patient_id"] + outcome_cols)]

    train_df, test_df = train_test_split(
        data, test_size=0.15, random_state=42, stratify=data["any_def_12m"]
    )

    X_train = train_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()
    y_train = train_df[task].astype(int).to_numpy()
    y_test  = test_df[task].astype(int).to_numpy()

    pre_tree = make_preprocessor(X_train, scale_numeric=False)
    pre_svm  = make_preprocessor(X_train, scale_numeric=True)

    # Models
    lgbm = LGBMClassifier(
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9, random_state=42
    )
    xgb = XGBClassifier(
        n_estimators=800, learning_rate=0.03, max_depth=4,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, eval_metric="logloss"
    )
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42)

    models = {
        "LightGBM": (pre_tree, lgbm),
        "XGBoost":  (pre_tree, xgb),
        "SVM":      (pre_svm, svm),
    }

    train_acc = {}
    test_acc = {}

    for name, (pre, base_model) in models.items():
        pipe = Pipeline([("pre", pre), ("model", base_model)])
        cal = CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
        cal.fit(X_train, y_train)

        # Use 0.5 threshold internally (you can avoid mentioning it in thesis)
        proba_train = cal.predict_proba(X_train)[:, 1]
        proba_test  = cal.predict_proba(X_test)[:, 1]
        yhat_train = (proba_train >= 0.5).astype(int)
        yhat_test  = (proba_test  >= 0.5).astype(int)

        train_acc[name] = float((yhat_train == y_train).mean())
        test_acc[name]  = float((yhat_test  == y_test).mean())

        cm = confusion_matrix(y_test, yhat_test)
        plot_confusion(cm, f"{name} – Confusion Matrix ({task})", f"{name}_{task}_cm.png")

    # 1st plot: Training -> Testing accuracy comparison
    plt.figure(figsize=(7,4))
    x = np.array([0, 1])
    for name in models.keys():
        plt.plot(x, [train_acc[name]*100, test_acc[name]*100], marker="o", label=name)

    plt.xticks([0,1], ["Training", "Testing"])
    plt.ylabel("Aniqlik (%)")
    plt.xlabel("Model Bosqichi")
    plt.title(f"ML Modellarning Training → Testing o‘zgarishi (Solishtirma) – {task}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"train_test_accuracy_{task}.png", dpi=200)
    plt.close()

    print("Saved images:")
    print(f"- train_test_accuracy_{task}.png")
    for name in models.keys():
        print(f"- {name}_{task}_cm.png")

if __name__ == "__main__":
    main(task="any_def_12m")  # xohlasangiz: "readmission_30d" yoki "complication_90d"