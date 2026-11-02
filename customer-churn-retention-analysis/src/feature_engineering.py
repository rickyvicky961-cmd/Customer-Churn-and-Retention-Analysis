"""
feature_engineering.py
----------------------
Module for feature engineering, transformation pipelines, and feature selection.
Constructs scikit-learn ColumnTransformer objects ensuring zero data leakage.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class CustomerFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible Transformer to engineer domain-specific churn features:
    - TotalServices: count of active service subscriptions
    - HasTechSupportOrSecurity: flag for high-retention services
    - IsAutomaticPayment: flag for auto bank transfer or credit card
    - AvgMonthlyChargesRatio: TotalCharges / (tenure + 1)
    - FiberWithoutSupport: indicator for high-risk fiber customers without tech support
    """
    def __init__(self):
        self.service_cols = [
            "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        if isinstance(X_out, np.ndarray):
            return X_out
            
        # Count active services
        services_count = pd.Series(0, index=X_out.index)
        for col in self.service_cols:
            if col in X_out.columns:
                services_count += (X_out[col] == "Yes").astype(int)
        X_out["TotalServices"] = services_count
        
        # Key retention protective factor
        sec = (X_out["OnlineSecurity"] == "Yes") if "OnlineSecurity" in X_out.columns else False
        tech = (X_out["TechSupport"] == "Yes") if "TechSupport" in X_out.columns else False
        X_out["HasTechSupportOrSecurity"] = (sec | tech).astype(int)
        
        # Payment automation
        if "PaymentMethod" in X_out.columns:
            X_out["IsAutomaticPayment"] = X_out["PaymentMethod"].isin([
                "Bank transfer (automatic)", "Credit card (automatic)"
            ]).astype(int)
        else:
            X_out["IsAutomaticPayment"] = 0
            
        # Charge ratios
        if "TotalCharges" in X_out.columns and "tenure" in X_out.columns:
            X_out["AvgMonthlyChargesRatio"] = X_out["TotalCharges"] / (X_out["tenure"] + 1)
        
        # High risk fiber user without support
        if "InternetService" in X_out.columns and "TechSupport" in X_out.columns:
            X_out["FiberWithoutSupport"] = (
                (X_out["InternetService"] == "Fiber optic") & 
                (X_out["TechSupport"] == "No")
            ).astype(int)
            
        return X_out


def get_feature_lists() -> Tuple[List[str], List[str]]:
    """
    Return lists of numerical and categorical feature column names.
    """
    numerical_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "TotalServices",
        "HasTechSupportOrSecurity",
        "IsAutomaticPayment",
        "AvgMonthlyChargesRatio",
        "FiberWithoutSupport"
    ]
    
    categorical_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]
    
    return numerical_features, categorical_features


def build_preprocessor() -> ColumnTransformer:
    """
    Construct Scikit-Learn ColumnTransformer for numerical scaling and categorical encoding.
    Ensures safe handling of unknown categories and zero data leakage.
    """
    num_cols, cat_cols = get_feature_lists()
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ],
        remainder="drop"
    )
    
    return preprocessor


def engineer_dataframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience helper to apply feature engineering on a DataFrame directly."""
    fe = CustomerFeatureEngineer()
    return fe.transform(df)


if __name__ == "__main__":
    from src.data_cleaning import load_raw_data, clean_customer_data
    raw = load_raw_data()
    clean, _ = clean_customer_data(raw)
    engineered = engineer_dataframe_features(clean)
    print(f"Feature engineering successful. Engineered shape: {engineered.shape}")
    print("New columns:", [c for c in engineered.columns if c not in clean.columns])

