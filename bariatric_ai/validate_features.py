#!/usr/bin/env python3
"""
Diagnostic script to validate feature columns and detect potential data leakage.

This script prints the final feature columns used in the evaluation pipeline
and highlights any suspicious columns that might indicate data leakage.
"""

import sys
import re
import pandas as pd
from evaluate_models import (
    build_features, 
    validate_feature_columns,
    FORBIDDEN_FEATURE_PATTERNS,
    LONG_PATH,
    OUT_PATH,
)


def print_feature_analysis():
    """Print detailed analysis of features used in the model."""
    
    print("="*80)
    print("FEATURE VALIDATION DIAGNOSTIC")
    print("="*80)
    print()
    
    # Load data
    print("Loading data...")
    long_df = pd.read_csv(LONG_PATH)
    outcomes = pd.read_csv(OUT_PATH)
    print(f"  Loaded {len(long_df)} rows from longitudinal data")
    print(f"  Loaded {len(outcomes)} patients from outcomes data")
    print()
    
    # Build features
    print("Building features from discharge + early (2w/1m) timepoints...")
    X = build_features(long_df)
    print(f"  Built features for {len(X)} patients")
    print()
    
    # Merge outcomes
    data = X.merge(outcomes, on="patient_id", how="inner")
    print(f"After merge: {len(data)} patients with {len(data.columns)} total columns")
    print()
    
    # Define exclusions
    targets_clf = ["readmission_30d", "complication_90d", "any_def_12m"]
    target_reg = "twl_12m"
    label_cols = ["iron_def_12m", "b12_def_12m", "vitd_def_12m"]
    
    all_excluded = ["patient_id"] + targets_clf + [target_reg] + label_cols
    
    print("Excluded columns (targets + labels):")
    for col in sorted(all_excluded):
        print(f"  - {col}")
    print()
    
    # Select features
    feature_cols = [c for c in data.columns if c not in all_excluded]
    
    print(f"Total feature columns: {len(feature_cols)}")
    print()
    
    # Check for suspicious patterns
    print("Checking for forbidden patterns...")
    suspicious = []
    for col in feature_cols:
        for pattern in FORBIDDEN_FEATURE_PATTERNS:
            if re.match(pattern, col, re.IGNORECASE):
                suspicious.append((col, pattern))
                break
    
    if suspicious:
        print("\n" + "!"*80)
        print("WARNING: SUSPICIOUS COLUMNS FOUND!")
        print("!"*80)
        for col, pattern in suspicious:
            print(f"  - '{col}' matches forbidden pattern '{pattern}'")
        print()
        print("These columns may indicate data leakage!")
        print("!"*80)
        print()
        return_code = 1
    else:
        print("✓ No forbidden patterns detected")
        print()
        return_code = 0
    
    # Print all features
    print("="*80)
    print("COMPLETE FEATURE LIST")
    print("="*80)
    for i, col in enumerate(sorted(feature_cols), 1):
        marker = " ⚠️ " if any(re.match(p, col, re.IGNORECASE) for p in FORBIDDEN_FEATURE_PATTERNS) else "   "
        print(f"{marker}{i:3d}. {col}")
    print()
    
    # Categorize features
    discharge_features = [c for c in feature_cols if not c.startswith("early_")]
    early_features = [c for c in feature_cols if c.startswith("early_")]
    
    print("="*80)
    print("FEATURE SUMMARY")
    print("="*80)
    print(f"Discharge features: {len(discharge_features)}")
    print(f"Early aggregates (2w/1m): {len(early_features)}")
    print(f"Total: {len(feature_cols)}")
    print()
    
    # Validate using the built-in validator
    print("Running validation...")
    try:
        validate_feature_columns(feature_cols)
        print("✓ VALIDATION PASSED: No data leakage detected")
        print()
    except ValueError as e:
        print("✗ VALIDATION FAILED:")
        print(str(e))
        print()
        return_code = 1
    
    print("="*80)
    
    return return_code


if __name__ == "__main__":
    sys.exit(print_feature_analysis())
