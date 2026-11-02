"""
data_cleaning.py
----------------
Module for loading, cleaning, validating, and preprocessing customer churn data.
Provides a reusable data cleaning pipeline that handles missing values,
type conversions, duplicate removal, outlier analysis, and database persistence.
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np


def get_project_root() -> Path:
    """Return the base customer-churn-retention-analysis project directory."""
    current = Path(__file__).resolve().parent
    if current.name == "src":
        return current.parent
    # Fallback search
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() or (parent / "customer-churn-retention-analysis").exists():
            if (parent / "data").exists():
                return parent
            return parent / "customer-churn-retention-analysis"
    return current


def get_raw_data_path() -> Path:
    """Resolve path to raw customer churn CSV."""
    root = get_project_root()
    return root / "data" / "raw" / "customer_churn.csv"


def get_processed_data_path() -> Path:
    """Resolve path to processed customer churn CSV."""
    root = get_project_root()
    return root / "data" / "processed" / "cleaned_customer_data.csv"


def get_database_path() -> Path:
    """Resolve path to SQLite database."""
    root = get_project_root()
    return root / "database" / "customer_database.db"


def load_raw_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw customer churn dataset from CSV.
    """
    path = Path(file_path) if file_path else get_raw_data_path()
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    
    df = pd.read_csv(path)
    return df


def detect_outliers_iqr(df: pd.DataFrame, columns: list = None) -> Dict[str, Dict[str, Any]]:
    """
    Detect outliers in numerical columns using the Interquartile Range (IQR) method.
    Returns summary statistics, lower and upper bounds, and count of outliers.
    """
    if columns is None:
        columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    
    outlier_report = {}
    for col in columns:
        if col in df.columns:
            # Ensure series is numeric
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = s[(s < lower_bound) | (s > upper_bound)]
            
            outlier_report[col] = {
                "Q1": round(float(q1), 2),
                "Q3": round(float(q3), 2),
                "IQR": round(float(iqr), 2),
                "lower_bound": round(float(lower_bound), 2),
                "upper_bound": round(float(upper_bound), 2),
                "outlier_count": int(len(outliers)),
                "outlier_percentage": round(float(len(outliers) / len(s) * 100), 2)
            }
    return outlier_report


def clean_customer_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute full end-to-end data cleaning workflow:
    1. Validate required columns.
    2. Check and remove duplicate rows.
    3. Correct TotalCharges data type (handle whitespace / NaN).
    4. Fill missing TotalCharges values for zero-tenure customers with 0.0 or MonthlyCharges.
    5. Clean string fields (strip leading/trailing whitespaces).
    6. Standardize SeniorCitizen as integer and categorical friendly.
    7. Standardize target 'Churn' to binary indicator 'Churn_Numeric' (1/0) while keeping 'Churn' ('Yes'/'No').
    
    Returns:
        cleaned_df: Cleaned pandas DataFrame.
        cleaning_stats: Dictionary containing diagnostic details of cleaning steps.
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    # 1. Remove duplicate records
    duplicate_count = int(df_clean.duplicated().sum())
    if duplicate_count > 0:
        df_clean = df_clean.drop_duplicates()
    
    # 2. Strip whitespace from string columns
    str_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    for c in str_cols:
        df_clean[c] = df_clean[c].astype(str).str.strip()
    
    # 3. Handle TotalCharges numeric conversion & missing values
    # In IBM Telco, blank string ' ' occurs when tenure == 0
    total_charges_raw = df_clean["TotalCharges"].replace(["", " ", "nan", "NaN", "null"], np.nan)
    total_charges_numeric = pd.to_numeric(total_charges_raw, errors="coerce")
    missing_total_charges = int(total_charges_numeric.isna().sum())
    
    # For new customers with tenure = 0, TotalCharges is legitimately 0.0
    # For others if any, impute with MonthlyCharges * tenure or median
    imputed_total_charges = np.where(
        total_charges_numeric.isna(),
        np.where(df_clean["tenure"] == 0, 0.0, df_clean["MonthlyCharges"] * df_clean["tenure"]),
        total_charges_numeric
    )
    df_clean["TotalCharges"] = np.round(imputed_total_charges.astype(float), 2)
    
    # 4. Correct data types
    df_clean["tenure"] = pd.to_numeric(df_clean["tenure"], errors="coerce").fillna(0).astype(int)
    df_clean["MonthlyCharges"] = pd.to_numeric(df_clean["MonthlyCharges"], errors="coerce").astype(float)
    df_clean["SeniorCitizen"] = pd.to_numeric(df_clean["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    
    # 5. Standardize target column Churn
    if "Churn" in df_clean.columns:
        df_clean["Churn"] = df_clean["Churn"].replace({"True": "Yes", "False": "No", "1": "Yes", "0": "No"})
        df_clean["Churn_Numeric"] = (df_clean["Churn"] == "Yes").astype(int)
    
    # 6. Outlier diagnostics
    outliers = detect_outliers_iqr(df_clean)
    
    cleaning_stats = {
        "initial_rows": initial_rows,
        "final_rows": len(df_clean),
        "duplicate_rows_removed": duplicate_count,
        "missing_total_charges_imputed": missing_total_charges,
        "columns_count": len(df_clean.columns),
        "outlier_summary": outliers
    }
    
    return df_clean, cleaning_stats


def save_to_database(df_raw: pd.DataFrame, df_cleaned: pd.DataFrame, db_path: Optional[str] = None) -> str:
    """
    Save raw and cleaned datasets to SQLite database tables for SQL analytics.
    """
    target_db = Path(db_path) if db_path else get_database_path()
    target_db.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(target_db))
    try:
        # Write tables
        df_raw.to_sql("customers_raw", conn, if_exists="replace", index=False)
        df_cleaned.to_sql("customers_cleaned", conn, if_exists="replace", index=False)
        
        # Create an index on customerID and Churn for fast querying
        with conn:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cleaned_cust_id ON customers_cleaned(customerID);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cleaned_churn ON customers_cleaned(Churn);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cleaned_contract ON customers_cleaned(Contract);")
            
        return str(target_db)
    finally:
        conn.close()


def run_data_cleaning_pipeline(
    raw_csv: Optional[str] = None,
    output_csv: Optional[str] = None,
    db_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main driver to run the complete data cleaning and persistence pipeline.
    """
    print("=" * 60)
    print("STARTING DATA CLEANING & PREPROCESSING PIPELINE")
    print("=" * 60)
    
    raw_df = load_raw_data(raw_csv)
    print(f"Loaded raw dataset successfully with shape: {raw_df.shape}")
    
    cleaned_df, stats = clean_customer_data(raw_df)
    print(f"Cleaned dataset: {stats['final_rows']} rows, {stats['columns_count']} columns.")
    print(f"Duplicates removed: {stats['duplicate_rows_removed']}")
    print(f"Missing TotalCharges values imputed: {stats['missing_total_charges_imputed']}")
    print(f"Outlier diagnostics: {stats['outlier_summary']}")
    
    # Save processed CSV
    out_path = Path(output_csv) if output_csv else get_processed_data_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(out_path, index=False)
    print(f"Saved processed dataset to: {out_path}")
    
    # Save to SQLite
    saved_db = save_to_database(raw_df, cleaned_df, db_path)
    print(f"Populated SQLite database at: {saved_db}")
    print("DATA CLEANING PIPELINE COMPLETED SUCCESSFULLY!\n")
    
    return cleaned_df, stats


if __name__ == "__main__":
    run_data_cleaning_pipeline()
