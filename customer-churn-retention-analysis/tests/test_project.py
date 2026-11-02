"""
test_project.py
---------------
Comprehensive test suite for Customer Churn and Retention Analysis project.
Tests data cleaning, feature engineering, model training/inference,
customer segmentation, SQLite persistence, and recommendation engine.
"""

import os
import sys
import sqlite3
import unittest
from pathlib import Path

# Add project base directory to sys.path
_current_dir = Path(__file__).resolve().parent
_base_dir = _current_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

import pandas as pd
import numpy as np

from src.data_cleaning import (
    load_raw_data, clean_customer_data, detect_outliers_iqr,
    get_project_root, get_raw_data_path, get_database_path
)
from src.feature_engineering import CustomerFeatureEngineer, build_preprocessor, engineer_dataframe_features
from src.exploratory_analysis import calculate_churn_kpis
from src.customer_segmentation import perform_customer_clustering
from src.retention_recommendations import (
    generate_personalized_retention_plan,
    answer_business_questions,
    analyze_customer_risk_drivers
)


class TestCustomerChurnProject(unittest.TestCase):
    """Unit and Integration tests for customer churn pipeline."""

    @classmethod
    def setUpClass(cls):
        """Load and clean dataset once for test efficiency."""
        cls.raw_df = load_raw_data()
        cls.clean_df, cls.clean_stats = clean_customer_data(cls.raw_df)
        cls.db_path = get_database_path()

    def test_01_raw_data_loading(self):
        """Verify raw data loads properly with minimum row count and expected columns."""
        self.assertGreaterEqual(len(self.raw_df), 5000, "Raw dataset should have at least 5000 rows.")
        expected_cols = [
            "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "PhoneService", "InternetService", "Contract",
            "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
        ]
        for col in expected_cols:
            self.assertIn(col, self.raw_df.columns, f"Missing required column: {col}")

    def test_02_data_cleaning_and_types(self):
        """Verify missing value imputation, TotalCharges numeric conversion, and no remaining NaNs."""
        self.assertEqual(len(self.clean_df), len(self.raw_df), "Cleaning should preserve valid rows.")
        self.assertTrue(pd.api.types.is_numeric_dtype(self.clean_df["TotalCharges"]), "TotalCharges must be numeric float.")
        self.assertTrue(pd.api.types.is_numeric_dtype(self.clean_df["tenure"]), "Tenure must be numeric int.")
        self.assertEqual(int(self.clean_df["TotalCharges"].isna().sum()), 0, "No NaNs should exist in TotalCharges.")
        self.assertIn("Churn_Numeric", self.clean_df.columns, "Churn_Numeric (1/0) indicator must exist.")
        self.assertTrue(set(self.clean_df["Churn_Numeric"].unique()).issubset({0, 1}))

    def test_03_outlier_detection(self):
        """Verify IQR outlier detection function executes and returns valid structure."""
        outliers = detect_outliers_iqr(self.clean_df, ["tenure", "MonthlyCharges", "TotalCharges"])
        self.assertIn("tenure", outliers)
        self.assertIn("MonthlyCharges", outliers)
        self.assertIn("TotalCharges", outliers)
        for col, stats in outliers.items():
            self.assertIn("IQR", stats)
            self.assertIn("lower_bound", stats)
            self.assertIn("upper_bound", stats)
            self.assertIn("outlier_count", stats)

    def test_04_feature_engineering_transformer(self):
        """Verify CustomerFeatureEngineer creates all expected domain-specific features."""
        fe = CustomerFeatureEngineer()
        engineered = fe.transform(self.clean_df)
        
        expected_new_cols = [
            "TotalServices", "HasTechSupportOrSecurity",
            "IsAutomaticPayment", "AvgMonthlyChargesRatio", "FiberWithoutSupport"
        ]
        for col in expected_new_cols:
            self.assertIn(col, engineered.columns, f"Engineered column {col} missing.")
            
        # TotalServices must be an integer between 0 and 8
        self.assertTrue(engineered["TotalServices"].between(0, 8).all(), "TotalServices must be between 0 and 8.")
        self.assertTrue(set(engineered["IsAutomaticPayment"].unique()).issubset({0, 1}))

    def test_05_preprocessor_pipeline(self):
        """Verify Scikit-learn preprocessor ColumnTransformer fits and transforms without leakage."""
        fe = CustomerFeatureEngineer()
        eng_df = fe.transform(self.clean_df)
        preprocessor = build_preprocessor()
        
        X = eng_df.drop(columns=["customerID", "Churn", "Churn_Numeric"], errors="ignore")
        transformed = preprocessor.fit_transform(X)
        self.assertIsInstance(transformed, np.ndarray)
        self.assertEqual(transformed.shape[0], len(self.clean_df))
        self.assertGreater(transformed.shape[1], 15, "Transformed features must be one-hot expanded.")

    def test_06_kpis_calculation(self):
        """Verify churn KPIs calculation logic and bounds."""
        kpis = calculate_churn_kpis(self.clean_df)
        self.assertEqual(kpis["total_customers"], len(self.clean_df))
        self.assertAlmostEqual(kpis["churn_rate_pct"] + kpis["retention_rate_pct"], 100.0, delta=0.1)
        self.assertGreater(kpis["avg_customer_tenure_months"], 0)
        self.assertGreater(kpis["avg_monthly_charges"], 0)
        self.assertGreater(kpis["total_annual_revenue_lost"], 0)

    def test_07_customer_segmentation_kmeans(self):
        """Verify K-Means clustering generates 4 named segments with proper metadata."""
        df_seg, summary, kmeans, scaler = perform_customer_clustering(self.clean_df, k=4)
        self.assertIn("Segment_Name", df_seg.columns)
        self.assertEqual(len(summary), 4)
        expected_segments = {
            "High-Value Loyal", "New & Vulnerable",
            "High-Risk High-Spenders", "Low-Engagement / Budget"
        }
        self.assertEqual(set(summary["Segment_Name"].unique()), expected_segments)

    def test_08_sqlite_database_persistence(self):
        """Verify SQLite database exists and contains expected tables with correct data."""
        self.assertTrue(self.db_path.exists(), f"Database not found at {self.db_path}")
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ["customers_raw", "customers_cleaned", "churn_kpis"]
            for table in required_tables:
                self.assertIn(table, tables, f"Expected table {table} missing from SQLite.")
                
            # Verify row counts
            cnt_raw = pd.read_sql_query("SELECT COUNT(*) as c FROM customers_raw", conn).iloc[0]["c"]
            cnt_clean = pd.read_sql_query("SELECT COUNT(*) as c FROM customers_cleaned", conn).iloc[0]["c"]
            self.assertEqual(cnt_raw, len(self.raw_df))
            self.assertEqual(cnt_clean, len(self.clean_df))
        finally:
            conn.close()

    def test_09_retention_recommendations_engine(self):
        """Verify personalized retention plan generation for a high-risk profile."""
        sample_high_risk = {
            "customerID": "TEST-RISK-01",
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "TechSupport": "No",
            "OnlineSecurity": "No",
            "PaymentMethod": "Electronic check",
            "tenure": 4,
            "MonthlyCharges": 89.90,
            "TotalCharges": 359.60
        }
        plan = generate_personalized_retention_plan(sample_high_risk, churn_prob=0.85)
        self.assertEqual(plan["risk_level"], "CRITICAL")
        self.assertGreater(len(plan["recommended_actions"]), 0)
        self.assertGreater(plan["revenue_at_risk_annual"], 0)
        self.assertGreater(plan["saved_revenue_potential"], 0)
        
        # Test business questions answer dictionary
        qa = answer_business_questions()
        self.assertEqual(len(qa), 10, "Must contain answers to all 10 business questions.")
        for k, v in qa.items():
            self.assertIn("question", v)
            self.assertIn("answer", v)
            self.assertGreater(len(v["answer"]), 20)


if __name__ == "__main__":
    unittest.main()

