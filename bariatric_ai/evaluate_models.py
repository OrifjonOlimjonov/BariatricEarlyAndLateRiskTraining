import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    brier_score_loss, log_loss, mean_absolute_error, r2_score,
    precision_score, recall_score
)
from sklearn.calibration import CalibratedClassifierCV

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Models
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.svm import SVC


LONG_PATH = "data_out/bariatric_long_10000_seed42.csv"
OUT_PATH  = "data_out/bariatric_outcomes_10000_seed42.csv"

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Forbidden feature patterns that indicate data leakage
FORBIDDEN_FEATURE_PATTERNS = [
    r".*_def_12m$",      # Deficiency labels at 12m
    r".*readmission.*",   # Readmission outcomes
    r".*complication.*",  # Complication outcomes
    r"^twl_12m$",        # Total weight loss at 12m (exact match)
    r".*_12m_true.*",    # Any underlying 12m truth values
]


TIMEPOINT_ORDER = {
    "discharge": 0,
    "2w": 1,
    "1m": 2,
    "3m": 3,
    "6m": 4,
    "12m": 5,
}


def _normalize_timepoint(tp):
    if tp in TIMEPOINT_ORDER:
        return tp
    s = str(tp).lower().strip()
    if s in ["2w", "2wk", "2wks", "2week", "2weeks"]:
        return "2w"
    if s in ["1m", "1mo", "1mon", "1month"]:
        return "1m"
    if s in ["3m", "3mo", "3mon", "3month"]:
        return "3m"
    if s in ["6m", "6mo", "6mon", "6month"]:
        return "6m"
    if s in ["12m", "12mo", "12mon", "12month", "1y", "1yr", "1year"]:
        return "12m"
    if s in ["discharge", "dc", "d/c"]:
        return "discharge"
    return tp


def validate_feature_columns(feature_cols: list) -> None:
    """
    Validate that feature columns do not contain forbidden patterns indicating data leakage.
    
    Raises:
        ValueError: If any forbidden column patterns are detected in feature_cols.
    """
    forbidden_found = []
    
    for col in feature_cols:
        for pattern in FORBIDDEN_FEATURE_PATTERNS:
            if re.match(pattern, col, re.IGNORECASE):
                forbidden_found.append((col, pattern))
                break
    
    if forbidden_found:
        error_msg = "LEAKAGE DETECTED: Forbidden columns found in features:\n"
        for col, pattern in forbidden_found:
            error_msg += f"  - '{col}' matches forbidden pattern '{pattern}'\n"
        error_msg += "\nThese columns contain future outcomes or labels and must be excluded from features."
        raise ValueError(error_msg)


def build_features(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()

    if "patient_id" not in df.columns:
        raise ValueError("long_df must include patient_id column")
    if "timepoint" not in df.columns:
        raise ValueError("long_df must include timepoint column")

    df["timepoint"] = df["timepoint"].apply(_normalize_timepoint)
    
    # Drop future outcome columns that should not be in features
    # twl_12m_true is the underlying 12m outcome - should not be available at discharge/early
    future_outcome_cols = ["twl_12m_true"]
    df = df.drop(columns=future_outcome_cols, errors="ignore")

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

    # Fill missing numeric early aggregates; keep categorical as-is
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


def find_best_threshold_f1(y_true: np.ndarray, y_proba: np.ndarray):
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_t, best_f1 = 0.5, -1.0
    best_prec, best_rec = 0.0, 0.0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
            best_prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
            best_rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))

    return best_t, best_prec, best_rec, float(best_f1)


def threshold_for_min_recall(y_true: np.ndarray, y_proba: np.ndarray, min_recall: float = 0.80):
    thresholds = np.linspace(0.0, 1.0, 1001)
    ok = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        if rec >= min_recall:
            prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            ok.append((float(t), float(prec), float(rec), float(f1)))
    if not ok:
        return np.nan, np.nan, np.nan, np.nan
    return ok[-1]


def eval_binary(y_true, y_proba, y_pred):
    out = {}
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_proba))
        out["auprc"] = float(average_precision_score(y_true, y_proba))
        out["brier"] = float(brier_score_loss(y_true, y_proba))
        out["logloss"] = float(log_loss(y_true, y_proba, labels=[0, 1]))
    else:
        out["auroc"] = np.nan
        out["auprc"] = np.nan
        out["brier"] = np.nan
        out["logloss"] = np.nan

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1_pos"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["pos_rate"] = float(np.mean(y_true))
    out["mean_proba"] = float(np.mean(y_proba))
    return out


def eval_regression(y_true, y_pred):
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))}


def compute_baseline_metrics(y_true: np.ndarray) -> dict:
    """
    Compute baseline metrics for a binary classification task.
    Baseline predicts the majority class for all samples.
    
    Returns:
        dict: Baseline accuracy and F1 scores
    """
    majority_class = 1 if np.mean(y_true) >= 0.5 else 0
    y_pred_baseline = np.full_like(y_true, majority_class)
    
    return {
        "baseline_accuracy": float(accuracy_score(y_true, y_pred_baseline)),
        "baseline_f1_pos": float(f1_score(y_true, y_pred_baseline, pos_label=1, zero_division=0)),
        "baseline_f1_macro": float(f1_score(y_true, y_pred_baseline, average="macro", zero_division=0)),
    }


def add_train_test_simple_metrics(row: dict, y_train: np.ndarray, proba_train: np.ndarray,
                                  y_test: np.ndarray, proba_test: np.ndarray, thr: float = 0.5):
    pred_train = (proba_train >= thr).astype(int)
    pred_test = (proba_test >= thr).astype(int)

    row.update({
        "train_accuracy_05": float(accuracy_score(y_train, pred_train)),
        "test_accuracy_05": float(accuracy_score(y_test, pred_test)),
        "train_f1_pos_05": float(f1_score(y_train, pred_train, pos_label=1, zero_division=0)),
        "test_f1_pos_05": float(f1_score(y_test, pred_test, pos_label=1, zero_division=0)),
    })


def main():
    long_df = pd.read_csv(LONG_PATH)
    outcomes = pd.read_csv(OUT_PATH)

    X = build_features(long_df)

    if "patient_id" not in outcomes.columns:
        raise ValueError("outcomes must include patient_id")
    
    # Define all target/label columns from outcomes that should NOT be features
    targets_clf = ["readmission_30d", "complication_90d", "any_def_12m"]
    target_reg = "twl_12m"
    # Additional label columns in outcomes that are NOT prediction targets but should be excluded
    label_cols = ["iron_def_12m", "b12_def_12m", "vitd_def_12m"]
    
    # Merge outcomes to get targets
    data = X.merge(outcomes, on="patient_id", how="inner")
    
    # Select features: exclude patient_id, all targets, and all label columns
    all_excluded = ["patient_id"] + targets_clf + [target_reg] + label_cols
    feature_cols = [c for c in data.columns if c not in all_excluded]
    
    # Validate no forbidden patterns in features (fail-fast on leakage)
    validate_feature_columns(feature_cols)
    
    print(f"\n{'='*80}")
    print(f"FEATURE VALIDATION PASSED: {len(feature_cols)} features")
    print(f"{'='*80}\n")
    
    # Train/test split stratified by patient_id (already done via random_state ensuring reproducibility)
    train_df, test_df = train_test_split(
        data, test_size=0.15, random_state=42, stratify=data["any_def_12m"]
    )

    X_train = train_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()

    pre_tree = make_preprocessor(X_train, scale_numeric=False)
    pre_svm = make_preprocessor(X_train, scale_numeric=True)

    results = []

    # --- Classification ---
    for ycol in targets_clf:
        y_train = train_df[ycol].astype(int).to_numpy()
        y_test  = test_df[ycol].astype(int).to_numpy()
        
        # Compute baseline metrics
        baseline_train = compute_baseline_metrics(y_train)
        baseline_test = compute_baseline_metrics(y_test)

        # --- LightGBM + calibration ---
        lgbm = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        lgbm_pipe = Pipeline([("pre", pre_tree), ("model", lgbm)])
        lgbm_cal = CalibratedClassifierCV(lgbm_pipe, method="sigmoid", cv=3)
        lgbm_cal.fit(X_train, y_train)

        proba_test = lgbm_cal.predict_proba(X_test)[:, 1]
        proba_train = lgbm_cal.predict_proba(X_train)[:, 1]
        pred05 = (proba_test >= 0.5).astype(int)

        row = {"model": "LightGBM_cal", "task": ycol, **eval_binary(y_test, proba_test, pred05)}
        add_train_test_simple_metrics(row, y_train, proba_train, y_test, proba_test, thr=0.5)
        row.update(baseline_test)  # Add baseline metrics for context

        bt, bp, br, bf1 = find_best_threshold_f1(y_test, proba_test)
        row.update({"best_thr_f1": bt, "best_thr_f1_precision": bp, "best_thr_f1_recall": br, "best_thr_f1_f1": bf1})

        t80, p80, r80, f180 = threshold_for_min_recall(y_test, proba_test, min_recall=0.80)
        row.update({"thr_recall80": t80, "thr_recall80_precision": p80, "thr_recall80_recall": r80, "thr_recall80_f1": f180})
        results.append(row)

        # --- XGBoost + calibration ---
        pos = float((y_train == 1).sum())
        neg = float((y_train == 0).sum())
        spw = (neg / pos) if pos > 0 else 1.0

        xgb = XGBClassifier(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=spw,
        )
        xgb_pipe = Pipeline([("pre", pre_tree), ("model", xgb)])
        xgb_cal = CalibratedClassifierCV(xgb_pipe, method="sigmoid", cv=3)
        xgb_cal.fit(X_train, y_train)

        proba_test = xgb_cal.predict_proba(X_test)[:, 1]
        proba_train = xgb_cal.predict_proba(X_train)[:, 1]
        pred05 = (proba_test >= 0.5).astype(int)

        row = {"model": "XGBoost_cal", "task": ycol, **eval_binary(y_test, proba_test, pred05)}
        add_train_test_simple_metrics(row, y_train, proba_train, y_test, proba_test, thr=0.5)
        row.update(baseline_test)  # Add baseline metrics for context

        bt, bp, br, bf1 = find_best_threshold_f1(y_test, proba_test)
        row.update({"best_thr_f1": bt, "best_thr_f1_precision": bp, "best_thr_f1_recall": br, "best_thr_f1_f1": bf1})

        t80, p80, r80, f180 = threshold_for_min_recall(y_test, proba_test, min_recall=0.80)
        row.update({"thr_recall80": t80, "thr_recall80_precision": p80, "thr_recall80_recall": r80, "thr_recall80_f1": f180})
        results.append(row)

        # --- SVM (RBF) + calibration ---
        svm = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=42,
        )
        svm_pipe = Pipeline([("pre", pre_svm), ("model", svm)])
        svm_cal = CalibratedClassifierCV(svm_pipe, method="sigmoid", cv=3)
        svm_cal.fit(X_train, y_train)

        proba_test = svm_cal.predict_proba(X_test)[:, 1]
        proba_train = svm_cal.predict_proba(X_train)[:, 1]
        pred05 = (proba_test >= 0.5).astype(int)

        row = {"model": "SVM_cal", "task": ycol, **eval_binary(y_test, proba_test, pred05)}
        add_train_test_simple_metrics(row, y_train, proba_train, y_test, proba_test, thr=0.5)
        row.update(baseline_test)  # Add baseline metrics for context

        bt, bp, br, bf1 = find_best_threshold_f1(y_test, proba_test)
        row.update({"best_thr_f1": bt, "best_thr_f1_precision": bp, "best_thr_f1_recall": br, "best_thr_f1_f1": bf1})

        t80, p80, r80, f180 = threshold_for_min_recall(y_test, proba_test, min_recall=0.80)
        row.update({"thr_recall80": t80, "thr_recall80_precision": p80, "thr_recall80_recall": r80, "thr_recall80_f1": f180})
        results.append(row)

    # --- Regression ---
    y_train_r = train_df[target_reg].astype(float)
    y_test_r  = test_df[target_reg].astype(float)

    lgbm_r = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    lgbm_r_pipe = Pipeline([("pre", pre_tree), ("model", lgbm_r)])
    lgbm_r_pipe.fit(X_train, y_train_r)
    pred = lgbm_r_pipe.predict(X_test)
    results.append({"model": "LightGBM", "task": target_reg, **eval_regression(y_test_r, pred)})

    xgb_r = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
    )
    xgb_r_pipe = Pipeline([("pre", pre_tree), ("model", xgb_r)])
    xgb_r_pipe.fit(X_train, y_train_r)
    pred = xgb_r_pipe.predict(X_test)
    results.append({"model": "XGBoost", "task": target_reg, **eval_regression(y_test_r, pred)})

    res_df = pd.DataFrame(results)
    out_csv = REPORT_DIR / "metrics_lgbm_vs_xgb.csv"
    res_df.to_csv(out_csv, index=False)

    print("Saved:", out_csv)
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()