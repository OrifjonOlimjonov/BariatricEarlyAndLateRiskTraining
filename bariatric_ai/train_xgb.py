import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, r2_score
import joblib

from xgboost import XGBClassifier, XGBRegressor

from data_generate import generate_synthetic_bariatric
from features import add_missingness, make_snapshot, make_early_features


def build_preprocessor(X: pd.DataFrame):
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop"
    )
    return pre


def eval_clf(name, pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "auroc": float(roc_auc_score(y_test, proba)),
        "auprc": float(average_precision_score(y_test, proba)),
        "pos_rate": float(np.mean(y_test)),
    }


def eval_reg(name, pipe, X_test, y_test):
    pred = pipe.predict(X_test)
    return {
        "model": name,
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }


def main():
    long_df, y = generate_synthetic_bariatric(n_patients=10_000, seed=42)
    long_df = add_missingness(long_df, seed=7)

    # discharge + early summary
    X0 = make_snapshot(long_df, "discharge")
    Xearly = make_early_features(long_df)
    X = X0.join(Xearly, how="left")

    data = X.join(y.set_index("patient_id"), how="inner").reset_index()

    # Split by patient
    train_idx, test_idx = train_test_split(data["patient_id"], test_size=0.15, random_state=123, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1765, random_state=123, shuffle=True)  # ~15% val

    train = data[data.patient_id.isin(train_idx)].copy()
    val = data[data.patient_id.isin(val_idx)].copy()
    test = data[data.patient_id.isin(test_idx)].copy()

    feature_cols = [c for c in data.columns if c not in ["patient_id",
                                                         "readmission_30d","complication_90d",
                                                         "iron_def_12m","b12_def_12m","vitd_def_12m","any_def_12m",
                                                         "twl_12m"]]
    X_train, X_test = train[feature_cols], test[feature_cols]

    # --- Classification tasks ---
    clf_tasks = ["readmission_30d", "complication_90d", "any_def_12m"]
    clf_results = []
    clf_pipes = {}

    for t in clf_tasks:
        y_train, y_test = train[t], test[t]
        pre = build_preprocessor(X_train)

        # Handle imbalance a bit (especially for readmission/complication)
        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0

        clf = XGBClassifier(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=123,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )

        pipe = Pipeline([("pre", pre), ("model", clf)])
        pipe.fit(X_train, y_train)

        clf_results.append(eval_clf(f"xgb_{t}", pipe, X_test, y_test))
        clf_pipes[t] = pipe

    # --- Regression task ---
    pre = build_preprocessor(X_train)
    reg = XGBRegressor(
        n_estimators=900,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=123,
        n_jobs=-1,
    )
    reg_pipe = Pipeline([("pre", pre), ("model", reg)])
    reg_pipe.fit(X_train, train["twl_12m"])
    reg_results = eval_reg("xgb_twl_12m", reg_pipe, X_test, test["twl_12m"])

    print("=== Classification Results (Test) ===")
    print(pd.DataFrame(clf_results).sort_values("auprc", ascending=False).to_string(index=False))
    print("\n=== Regression Results (Test) ===")
    print(pd.DataFrame([reg_results]).to_string(index=False))

    joblib.dump({
        "feature_cols": feature_cols,
        "classifiers": clf_pipes,
        "regressor": reg_pipe,
    }, "bariatric_xgb_bundle.joblib")

    print("\nSaved: bariatric_xgb_bundle.joblib")


if __name__ == "__main__":
    main()