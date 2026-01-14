# Understanding Model Evaluation Metrics

## Overview

This document explains how to interpret the evaluation metrics for bariatric surgery risk prediction models, particularly in the context of **imbalanced classification tasks**.

## The Challenge of Imbalanced Classes

In healthcare prediction tasks, we often face **class imbalance** where one outcome (e.g., "no complication") is much more common than another (e.g., "complication occurred"). This imbalance has important implications for metric interpretation.

### Example: Complication Prediction

If only 4% of patients experience complications:
- **Naive baseline**: Always predicting "no complication" achieves **96% accuracy**
- However, this model is useless because it never identifies at-risk patients!

## Key Metrics Explained

### 1. Baseline Metrics (for Context)

We compute baseline metrics assuming a "majority class" predictor:

- **`baseline_accuracy`**: Accuracy if we always predict the majority class
- **`baseline_f1_pos`**: F1 score for positive class with majority predictor (often 0)
- **`baseline_f1_macro`**: Macro-averaged F1 with majority predictor

**Purpose**: These provide context for evaluating model performance. A model should significantly outperform these baselines.

### 2. Discrimination Metrics

These measure how well the model separates positive from negative cases:

- **`auroc`** (Area Under ROC Curve): Probability that a random positive case ranks higher than a random negative case
  - Range: 0.5 (random) to 1.0 (perfect)
  - **Good**: > 0.70
  - **Excellent**: > 0.80
  
- **`auprc`** (Area Under Precision-Recall Curve): Especially important for imbalanced data
  - More informative than AUROC when positive class is rare
  - **Good**: Should exceed positive class prevalence
  - **Excellent**: > 0.50 for rare events

### 3. Calibration Metrics

These measure how well predicted probabilities match actual outcomes:

- **`brier`** (Brier Score): Mean squared error of probability predictions
  - Range: 0.0 (perfect) to 1.0 (worst)
  - **Good**: < 0.10 for typical medical tasks
  
- **`logloss`** (Log Loss): Penalizes confident wrong predictions
  - Lower is better
  - **Good**: < 0.30 for binary classification

### 4. Classification Metrics (at specific threshold)

These depend on the chosen decision threshold (commonly 0.5):

- **`accuracy`**: Overall correct predictions ÷ total predictions
  - ⚠️ **Misleading with imbalanced data** - can be high even with poor model
  
- **`f1_pos`**: Harmonic mean of precision and recall for positive class
  - **Better for imbalanced data** than accuracy
  - Range: 0.0 to 1.0
  
- **`f1_macro`**: Average F1 across both classes
  - Treats both classes equally regardless of frequency

### 5. Threshold Optimization Metrics

Models output probabilities; we choose a threshold to make binary decisions:

- **`best_thr_f1`**: Threshold that maximizes F1 score
  - Often differs from 0.5 in imbalanced settings
  
- **`thr_recall80`**: Threshold that achieves ≥80% recall (sensitivity)
  - Useful for screening tasks where we want to catch most positive cases
  - Trade-off: Lower precision (more false positives)

## Why Accuracy Can Be Misleading

### Example: Deficiency Prediction

Suppose 60% of patients develop vitamin deficiency at 12 months.

**Scenario 1: Leaked Model** (has access to future lab values)
- Accuracy: 100%
- F1: 1.0
- AUROC: 1.0
- **Problem**: Impossible in practice - indicates data leakage!

**Scenario 2: Good Model** (uses only discharge + early data)
- Accuracy: 75%
- F1: 0.73
- AUROC: 0.82
- **Interpretation**: Actually excellent performance given only early indicators

**Scenario 3: Baseline Model** (always predicts majority class)
- Accuracy: 60% (just predict "deficiency" for everyone)
- F1: 0.75 (actually not terrible for this prevalence)
- AUROC: 0.50 (no discrimination)

## Red Flags for Data Leakage

If you see these patterns, suspect data leakage:

1. **Perfect or near-perfect metrics** (AUROC = 1.0, accuracy = 1.0)
   - Real-world medical prediction is never perfect
   
2. **Train accuracy = 1.0** consistently
   - Models should have some training error
   
3. **Metrics far exceed baseline by unrealistic margins**
   - e.g., 100% accuracy when baseline is 60%

4. **Different tasks show vastly different performance**
   - e.g., `any_def_12m` has AUROC=1.0 but `readmission_30d` has AUROC=0.67
   - Suggests first task has access to leaked information

## Best Practices

1. **Always compare to baseline** - Is the model better than a naive predictor?

2. **Focus on AUROC/AUPRC** for imbalanced data, not accuracy

3. **Consider clinical utility** - What threshold optimizes for the specific use case?
   - Screening: Maximize recall (catch all at-risk patients)
   - Confirmation: Maximize precision (minimize false alarms)

4. **Validate feature integrity**:
   - Features should only use data available at prediction time
   - No future outcomes or labels should be in feature set
   - Run `validate_features.py` to check for leakage

5. **Check train vs test performance**:
   - Large gaps suggest overfitting
   - Identical perfect scores suggest leakage

## Recommended Threshold Selection

For clinical deployment, consider these approaches:

### Screening/Early Detection
- **Goal**: Catch most at-risk patients (high recall)
- **Use**: `thr_recall80` threshold (or higher recall target)
- **Trade-off**: More false positives, but don't miss true cases

### Resource Allocation
- **Goal**: Balance detection with resource constraints
- **Use**: `best_thr_f1` threshold
- **Trade-off**: Optimizes precision-recall balance

### Confirmation/Intervention
- **Goal**: Minimize false positives (high precision)
- **Use**: Higher threshold than 0.5
- **Trade-off**: May miss some true positives

## Summary

| Metric | Good for Imbalanced? | Interpretation |
|--------|---------------------|----------------|
| Accuracy | ❌ No | Can be misleading; compare to baseline |
| AUROC | ✅ Yes | Overall discrimination ability |
| AUPRC | ✅✅ Best | Especially good for rare events |
| F1 Score | ✅ Yes | Balance of precision and recall |
| Brier/LogLoss | ✅ Yes | Calibration quality |

**Bottom line**: For imbalanced medical prediction tasks, prioritize **AUROC**, **AUPRC**, and **F1 score** over accuracy. Always validate against baseline and check for data leakage.
