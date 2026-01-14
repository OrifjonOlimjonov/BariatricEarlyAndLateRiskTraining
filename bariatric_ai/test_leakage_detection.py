#!/usr/bin/env python3
"""
Unit tests for data leakage detection and feature validation.

Tests the validate_feature_columns function and build_features to ensure
no forbidden columns are included in the feature set.
"""

import unittest
import pandas as pd
import numpy as np
from evaluate_models import (
    validate_feature_columns,
    build_features,
    FORBIDDEN_FEATURE_PATTERNS,
)


class TestLeakageDetection(unittest.TestCase):
    """Test suite for data leakage detection."""
    
    def test_validate_feature_columns_passes_with_clean_features(self):
        """Test that validation passes with legitimate features."""
        clean_features = [
            "age",
            "sex",
            "bmi",
            "early_weight_kg_mean",
            "early_albumin_max",
            "ferritin",
            "discharge_symptom",
        ]
        # Should not raise
        try:
            validate_feature_columns(clean_features)
        except ValueError:
            self.fail("validate_feature_columns raised ValueError unexpectedly!")
    
    def test_validate_feature_columns_detects_def_12m_pattern(self):
        """Test that validation detects *_def_12m patterns."""
        leaky_features = ["age", "bmi", "iron_def_12m", "weight_kg"]
        with self.assertRaises(ValueError) as context:
            validate_feature_columns(leaky_features)
        self.assertIn("iron_def_12m", str(context.exception))
        self.assertIn("LEAKAGE DETECTED", str(context.exception))
    
    def test_validate_feature_columns_detects_multiple_violations(self):
        """Test that validation detects multiple forbidden columns."""
        leaky_features = [
            "age",
            "b12_def_12m",
            "vitd_def_12m",
            "readmission_30d",
            "weight_kg"
        ]
        with self.assertRaises(ValueError) as context:
            validate_feature_columns(leaky_features)
        error_msg = str(context.exception)
        # Should mention multiple violations
        self.assertIn("b12_def_12m", error_msg)
        self.assertIn("vitd_def_12m", error_msg)
        self.assertIn("readmission", error_msg)
    
    def test_validate_feature_columns_detects_twl_12m(self):
        """Test that validation detects exact twl_12m pattern."""
        leaky_features = ["age", "twl_12m", "ferritin"]
        with self.assertRaises(ValueError) as context:
            validate_feature_columns(leaky_features)
        self.assertIn("twl_12m", str(context.exception))
    
    def test_validate_feature_columns_detects_twl_12m_true_pattern(self):
        """Test that validation detects *_12m_true* patterns."""
        leaky_features = ["age", "early_twl_12m_true_mean", "ferritin"]
        with self.assertRaises(ValueError) as context:
            validate_feature_columns(leaky_features)
        self.assertIn("early_twl_12m_true_mean", str(context.exception))
    
    def test_validate_feature_columns_detects_complication_pattern(self):
        """Test that validation detects complication patterns."""
        leaky_features = ["age", "complication_90d", "ferritin"]
        with self.assertRaises(ValueError) as context:
            validate_feature_columns(leaky_features)
        self.assertIn("complication", str(context.exception))
    
    def test_validate_feature_columns_detects_readmission_pattern(self):
        """Test that validation detects readmission patterns."""
        leaky_features = ["age", "readmission_indicator", "ferritin"]
        with self.assertRaises(ValueError) as context:
            validate_feature_columns(leaky_features)
        self.assertIn("readmission", str(context.exception))
    
    def test_build_features_excludes_twl_12m_true(self):
        """Test that build_features removes twl_12m_true column."""
        # Create minimal synthetic long_df
        long_df = pd.DataFrame({
            "patient_id": [1, 1, 1],
            "timepoint": ["discharge", "2w", "1m"],
            "age": [45, 45, 45],
            "weight_kg": [120, 115, 110],
            "twl_12m_true": [0.25, 0.25, 0.25],  # This should be excluded
        })
        
        features = build_features(long_df)
        
        # twl_12m_true should not be in any column
        self.assertNotIn("twl_12m_true", features.columns)
        # Also check early aggregates don't include it
        twl_cols = [c for c in features.columns if "twl_12m_true" in c.lower()]
        self.assertEqual(len(twl_cols), 0, 
                        f"Found forbidden columns: {twl_cols}")
    
    def test_build_features_includes_discharge_and_early(self):
        """Test that build_features correctly uses discharge + early data."""
        long_df = pd.DataFrame({
            "patient_id": [1, 1, 1, 1],
            "timepoint": ["discharge", "2w", "1m", "12m"],
            "age": [45, 45, 45, 45],
            "weight_kg": [120, 115, 110, 95],
            "ferritin": [80, 75, 70, 40],  # 12m value should not influence early aggregates
        })
        
        features = build_features(long_df)
        
        # Should have patient_id
        self.assertIn("patient_id", features.columns)
        # Should have discharge age
        self.assertIn("age", features.columns)
        self.assertEqual(features["age"].iloc[0], 45)
        # Should have discharge weight
        self.assertIn("weight_kg", features.columns)
        self.assertEqual(features["weight_kg"].iloc[0], 120)
        # Should have early aggregates from 2w and 1m only
        self.assertIn("early_weight_kg_mean", features.columns)
        self.assertIn("early_ferritin_mean", features.columns)
        # early_ferritin_mean should be (75+70)/2 = 72.5, not include 12m value of 40
        self.assertAlmostEqual(features["early_ferritin_mean"].iloc[0], 72.5, places=1)
    
    def test_forbidden_patterns_comprehensive(self):
        """Test that FORBIDDEN_FEATURE_PATTERNS covers expected cases."""
        import re
        
        test_cases = [
            ("iron_def_12m", True),
            ("b12_def_12m", True),
            ("vitd_def_12m", True),
            ("any_def_12m", True),
            ("readmission_30d", True),
            ("readmission_indicator", True),
            ("complication_90d", True),
            ("complication_flag", True),
            ("twl_12m", True),
            ("early_twl_12m_true_mean", True),
            ("some_12m_true_value", True),
            # These should NOT match (legitimate feature names)
            ("age", False),
            ("weight_kg", False),
            ("early_weight_kg_mean", False),
            ("ferritin", False),
            ("early_ferritin_12w", False),  # 12w (weeks), not caught by _12m or _12m_true patterns
        ]
        
        for column, should_match in test_cases:
            matches = any(re.match(p, column, re.IGNORECASE) 
                         for p in FORBIDDEN_FEATURE_PATTERNS)
            self.assertEqual(matches, should_match,
                           f"Column '{column}' should{'' if should_match else ' NOT'} match forbidden patterns")


class TestFeatureBuilding(unittest.TestCase):
    """Test suite for feature building logic."""
    
    def test_build_features_requires_patient_id(self):
        """Test that build_features requires patient_id column."""
        long_df = pd.DataFrame({
            "timepoint": ["discharge"],
            "age": [45],
        })
        with self.assertRaises(ValueError) as context:
            build_features(long_df)
        self.assertIn("patient_id", str(context.exception))
    
    def test_build_features_requires_timepoint(self):
        """Test that build_features requires timepoint column."""
        long_df = pd.DataFrame({
            "patient_id": [1],
            "age": [45],
        })
        with self.assertRaises(ValueError) as context:
            build_features(long_df)
        self.assertIn("timepoint", str(context.exception))
    
    def test_build_features_handles_missing_early_data(self):
        """Test that build_features handles patients with only discharge data."""
        long_df = pd.DataFrame({
            "patient_id": [1],
            "timepoint": ["discharge"],
            "age": [45],
            "weight_kg": [120],
        })
        
        features = build_features(long_df)
        
        # Should return one patient
        self.assertEqual(len(features), 1)
        # Should have discharge features
        self.assertEqual(features["age"].iloc[0], 45)
        # Early aggregates should be NaN and then filled with median
        self.assertIn("early_weight_kg_mean", features.columns)


if __name__ == "__main__":
    unittest.main()
