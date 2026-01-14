# Bariatric Early and Late Risk Training

Machine learning models for predicting post-bariatric surgery outcomes using discharge and early follow-up data.

## Overview

This repository contains code for training and evaluating risk prediction models for bariatric surgery patients. The models predict:

1. **30-day readmission** (`readmission_30d`)
2. **90-day complications** (`complication_90d`)  
3. **12-month nutritional deficiencies** (`any_def_12m`, including iron, B12, and vitamin D)
4. **12-month total weight loss** (`twl_12m`) - regression task

## Key Features

### Data Leakage Prevention

The evaluation pipeline includes robust safeguards to prevent data leakage:

- **Feature Validation**: Automatically detects and blocks forbidden columns that contain future outcomes
- **Temporal Constraints**: Only uses data from discharge + early follow-up (2 weeks, 1 month)
- **Pattern Matching**: Prevents inclusion of columns matching `*_def_12m`, `*readmission*`, `*complication*`, `twl_12m`

### Evaluation Metrics

Models are evaluated with metrics appropriate for imbalanced classification:

- **AUROC/AUPRC**: Primary metrics for discrimination
- **Baseline Comparisons**: Majority-class baseline for context
- **Calibration**: Brier score and log loss
- **Threshold Optimization**: Best F1 threshold and recall@80% threshold

See [METRICS_GUIDE.md](METRICS_GUIDE.md) for detailed explanation of metrics and interpretation.

## Usage

### 1. Feature Validation

Before training, validate that features don't contain data leakage:

```bash
cd bariatric_ai
python validate_features.py
```

This script:
- Loads the longitudinal data and outcomes
- Builds features using only discharge + early (2w/1m) timepoints
- Validates no forbidden columns are present
- Prints complete feature list with categorization

**Expected output:**
```
✓ VALIDATION PASSED: No data leakage detected
```

### 2. Model Evaluation

Run the full evaluation pipeline:

```bash
cd bariatric_ai
python evaluate_models.py
```

This will:
- Build features from `data_out/bariatric_long_10000_seed42.csv`
- Load outcomes from `data_out/bariatric_outcomes_10000_seed42.csv`
- Validate features for leakage
- Train LightGBM, XGBoost, and SVM models with calibration
- Save results to `reports/metrics_lgbm_vs_xgb.csv`

### 3. Run Tests

Test the leakage detection functionality:

```bash
cd bariatric_ai
python -m unittest test_leakage_detection -v
```

All tests should pass. The test suite validates:
- Forbidden pattern detection (e.g., `*_def_12m`, `*readmission*`)
- Feature building excludes future outcomes
- Build process uses only discharge + early data

## File Structure

```
bariatric_ai/
├── evaluate_models.py          # Main evaluation pipeline with leakage prevention
├── validate_features.py        # Diagnostic script for feature validation
├── test_leakage_detection.py   # Unit tests for leakage detection
├── features.py                 # Feature engineering utilities
├── data_generate.py            # Synthetic data generation
├── data_out/                   # Generated datasets
│   ├── bariatric_long_10000_seed42.csv
│   └── bariatric_outcomes_10000_seed42.csv
└── reports/                    # Evaluation results
    └── metrics_lgbm_vs_xgb.csv

METRICS_GUIDE.md                # Documentation on interpreting metrics
```

## Interpreting Results

### Understanding Perfect Metrics

**⚠️ Red Flags for Data Leakage:**

If you see these patterns, suspect data leakage:
- AUROC = 1.0 (perfect discrimination)
- Train accuracy = 1.0 consistently
- Test accuracy >> baseline by unrealistic margin

**Example of leaked results (BEFORE fix):**
```
Task: any_def_12m
AUROC: 1.0
Train Accuracy: 1.0
Test Accuracy: 1.0
```

**Example of fixed results (AFTER):**
```
Task: any_def_12m
AUROC: 0.91 (excellent, realistic)
Baseline Accuracy: 0.606 (majority class)
Test Accuracy: 0.80-0.84 (beats baseline significantly)
```

### Metrics to Focus On

For imbalanced classification tasks (most of our predictions):

1. **AUROC** - Overall discrimination ability
   - Good: > 0.70
   - Excellent: > 0.80

2. **AUPRC** - Especially important for rare events
   - Should exceed positive class prevalence
   - Good: > 0.50 for rare events

3. **Baseline Comparison** - Context for performance
   - Model should significantly outperform baseline
   - Baseline = always predicting majority class

4. **Don't rely on accuracy alone** - Can be misleading with imbalanced data

See [METRICS_GUIDE.md](METRICS_GUIDE.md) for comprehensive explanation.

## Feature Engineering

### Temporal Data Structure

The longitudinal dataset has one row per patient per timepoint:

**Timepoints:**
- `discharge` - Hospital discharge
- `2w` - 2 weeks post-op
- `1m` - 1 month post-op
- `3m` - 3 months post-op (not used for prediction)
- `6m` - 6 months post-op (not used for prediction)
- `12m` - 12 months post-op (not used for prediction)

### Feature Construction

Features are built from **discharge + early (2w, 1m) only**:

1. **Discharge snapshot**: Static features + discharge measurements
   - Demographics: age, sex, height
   - Pre-op: BMI, weight, comorbidities
   - Surgery: ASA score, operation duration, length of stay
   - Labs: HbA1c, ferritin, B12, vitamin D, albumin

2. **Early aggregates**: Statistics from 2w and 1m visits
   - Mean, min, max of all numeric measurements
   - Prefix: `early_*`
   - Example: `early_weight_kg_mean`, `early_ferritin_min`

### Forbidden Columns

The following patterns are **automatically blocked** from features:

- `*_def_12m` - Deficiency labels at 12 months
- `*readmission*` - Readmission outcomes
- `*complication*` - Complication outcomes  
- `twl_12m` - Total weight loss at 12 months
- `*_12m_true*` - Any underlying 12-month truth values

## Train/Test Split

- **Split method**: Random stratified split by `any_def_12m`
- **Test size**: 15%
- **Random seed**: 42 (reproducible)
- **Note**: Split is at patient level, not row level

## Models

All classification models use:
- Calibration via sigmoid (CalibratedClassifierCV with 3-fold CV)
- Tree-based models: LightGBM, XGBoost
- SVM: RBF kernel with class balancing

Regression models (for `twl_12m`):
- LightGBM Regressor
- XGBoost Regressor

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost lightgbm
```

## Citation

If you use this code, please cite:

```
Bariatric Early and Late Risk Training
Repository: OrifjonOlimjonov/BariatricEarlyAndLateRiskTraining
```

## License

[Add your license here]

## Contributing

When adding new features or models:

1. Run `validate_features.py` to check for leakage
2. Run `test_leakage_detection.py` to verify tests pass
3. Update `FORBIDDEN_FEATURE_PATTERNS` if adding new outcome columns
4. Document any new metrics or evaluation approaches

## Contact

[Add contact information]
