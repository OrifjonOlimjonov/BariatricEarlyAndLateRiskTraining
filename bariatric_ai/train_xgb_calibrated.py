import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    mean_absolute_error, r2_score, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
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


def eval_clf(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "auroc": float(roc_auc_score(y_test, proba)),
        "auprc": float(average_precision_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "pos_rate": float(np.mean(y_test)),
        "proba_mean": float(np.mean(proba)),
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

    # Split by patient (train/test only; calibration uses CV inside train)
    train_ids, test_ids = train_test_split(
        data["patient_id"], test_size=0.15, random_state=123, shuffle=True
    )
    train = data[data.patient_id.isin(train_ids)].copy()
    test = data[data.patient_id.isin(test_ids)].copy()

    feature_cols = [c for c in data.columns if c not in [
        "patient_id",
        "readmission_30d", "complication_90d",
        "iron_def_12m", "b12_def_12m", "vitd_def_12m", "any_def_12m",
        "twl_12m"
    ]]

    X_train, X_test = train[feature_cols], test[feature_cols]

    # ---- Classification tasks (CalibratedClassifierCV with cv folds) ----
    clf_tasks = ["readmission_30d", "complication_90d", "any_def_12m"]
    clf_results = []
    clf_models = {}

    for t in clf_tasks:
        y_train, y_test = train[t], test[t]

        # imbalance weight (train only)
        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0

        pre = build_preprocessor(X_train)
        base = XGBClassifier(
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

        base_pipe = Pipeline([("pre", pre), ("model", base)])

        # Calibrate using CV on TRAIN
        calibrated = CalibratedClassifierCV(
            estimator=base_pipe,
            method="sigmoid",
            cv=3,  # 3-fold calibration CV
        )
        calibrated.fit(X_train, y_train)

        clf_results.append(eval_clf(f"xgb_cal_{t}", calibrated, X_test, y_test))
        clf_models[t] = calibrated

    # ---- Regression task (not calibrated) ----
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

    print("=== Classification Results (Test, Calibrated) ===")
    print(pd.DataFrame(clf_results).sort_values("auprc", ascending=False).to_string(index=False))
    print("\n=== Regression Results (Test) ===")
    print(pd.DataFrame([reg_results]).to_string(index=False))

    out_path = "bariatric_xgb_calibrated_bundle.joblib"
    joblib.dump({
        "feature_cols": feature_cols,
        "classifiers": clf_models,
        "regressor": reg_pipe,
        "calibration": {"method": "sigmoid", "cv": 3},
    }, out_path)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()