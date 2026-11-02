"""
model_training.py
-----------------
Module for training, evaluating, and persisting Customer Churn Machine Learning models.
Trains and benchmarks:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, and ROC Curve.
Saves the best model bundle to models/churn_model.pkl.
"""

import os
import sys
import pickle
import sqlite3
from pathlib import Path

# Add project base directory to sys.path
_current_dir = Path(__file__).resolve().parent
_base_dir = _current_dir.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

from src.data_cleaning import load_raw_data, clean_customer_data, get_project_root
from src.feature_engineering import CustomerFeatureEngineer, build_preprocessor


def get_models_dict() -> Dict[str, Any]:
    """
    Define candidate machine learning classifiers configured for balanced learning.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            C=1.0,
            solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=15,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        )
    }


def train_and_evaluate_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame, Any, Dict[str, Any]]:
    """
    Train all 4 candidate models with full Scikit-Learn Pipelines (feature engineering + preprocessing).
    Prevents data leakage by strictly fitting transformers on X_train.
    
    Returns:
        results: Detailed metrics per model
        comparison_df: Tabular comparison of evaluation metrics
        best_pipeline: Best trained pipeline
        best_meta: Metadata of best model
    """
    X = df.drop(columns=["customerID", "Churn", "Churn_Numeric"], errors="ignore")
    y = df["Churn_Numeric"].values
    
    # Stratified Train-Test Split to preserve churn proportion
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    models = get_models_dict()
    results = {}
    rows = []
    
    best_score = -1.0
    best_model_name = None
    best_pipeline = None
    
    for name, clf in models.items():
        print(f"Training {name} ...")
        # Build complete leak-free Pipeline
        pipe = Pipeline(steps=[
            ("feature_engineer", CustomerFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("classifier", clf)
        ])
        
        # Train strictly on training split
        pipe.fit(X_train, y_train)
        
        # Inference on test set
        y_pred = pipe.predict(X_test)
        if hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        
        results[name] = {
            "pipeline": pipe,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": cm,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }
        
        rows.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4)
        })
        
        # Primary selection: Recall and ROC-AUC (prioritizing identification of churners)
        # Composite score weighting Recall (40%), ROC-AUC (40%), and F1 (20%)
        composite_score = 0.40 * rec + 0.40 * auc + 0.20 * f1
        if composite_score > best_score:
            best_score = composite_score
            best_model_name = name
            best_pipeline = pipe

    comparison_df = pd.DataFrame(rows).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    
    best_meta = {
        "best_model_name": best_model_name,
        "composite_score": round(float(best_score), 4),
        "metrics": results[best_model_name],
        "test_size": test_size,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    return results, comparison_df, best_pipeline, best_meta


def plot_model_evaluations(results: Dict[str, Any], comparison_df: pd.DataFrame, output_dir: Path) -> str:
    """
    Generate ROC curves and metrics comparison plot saved to visualizations/.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. ROC Curves
    for name, res in results.items():
        axes[0].plot(res["fpr"], res["tpr"], label=f"{name} (AUC = {res['roc_auc']:.3f})", linewidth=2)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess")
    axes[0].set_title("ROC Curves Comparison", fontweight="bold", pad=12)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")
    
    # 2. Bar Chart Comparison
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    df_plot = comparison_df.set_index("Model")[metric_cols]
    df_plot.plot(kind="bar", ax=axes[1], width=0.8, colormap="tab10", edgecolor="black")
    axes[1].set_title("Model Performance Benchmark Across Metrics", fontweight="bold", pad=12)
    axes[1].set_ylabel("Score (0.0 - 1.0)")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15, ha="right")
    axes[1].legend(loc="lower right", ncol=2)
    
    plt.suptitle("Machine Learning Churn Prediction Benchmark", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    out_file = output_dir / "model_roc_curves.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out_file)


def save_best_model(
    pipeline: Any,
    best_meta: Dict[str, Any],
    comparison_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> str:
    """
    Save best model pipeline and training metadata bundle using pickle.
    """
    root = get_project_root()
    path = Path(output_path) if output_path else root / "models" / "churn_model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    bundle = {
        "pipeline": pipeline,
        "best_model_name": best_meta["best_model_name"],
        "metrics": best_meta["metrics"],
        "comparison_table": comparison_df.to_dict(orient="records"),
        "train_samples": best_meta["train_samples"],
        "test_samples": best_meta["test_samples"]
    }
    
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
        
    return str(path)


def populate_predictions_in_database(
    pipeline: Any,
    df: pd.DataFrame,
    db_path: Optional[str] = None
) -> str:
    """
    Generate churn probability and risk tier for every customer and save to SQLite.
    """
    root = get_project_root()
    target_db = Path(db_path) if db_path else root / "database" / "customer_database.db"
    
    X = df.drop(columns=["customerID", "Churn", "Churn_Numeric"], errors="ignore")
    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    
    df_preds = df[["customerID", "gender", "Contract", "MonthlyCharges", "TotalCharges", "tenure", "Churn"]].copy()
    df_preds["ChurnProbability"] = np.round(probabilities, 4)
    df_preds["PredictedChurn"] = np.where(predictions == 1, "Yes", "No")
    
    # Assign Risk Tiers
    conditions = [
        df_preds["ChurnProbability"] >= 0.70,
        df_preds["ChurnProbability"] >= 0.40,
        df_preds["ChurnProbability"] < 0.40
    ]
    tiers = ["High Risk", "Medium Risk", "Low Risk"]
    df_preds["RiskTier"] = np.select(conditions, tiers, default="Low Risk")
    
    conn = sqlite3.connect(str(target_db))
    try:
        df_preds.to_sql("churn_predictions", conn, if_exists="replace", index=False)
        with conn:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_risk ON churn_predictions(RiskTier);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_prob ON churn_predictions(ChurnProbability);")
    finally:
        conn.close()
        
    return str(target_db)


def run_model_training_pipeline() -> Tuple[pd.DataFrame, str]:
    """
    End-to-end execution of model training, evaluation, artifact export, and database sync.
    """
    root = get_project_root()
    processed_csv = root / "data" / "processed" / "cleaned_customer_data.csv"
    if not processed_csv.exists():
        from src.data_cleaning import run_data_cleaning_pipeline
        df, _ = run_data_cleaning_pipeline()
    else:
        df = pd.read_csv(processed_csv)
        
    print("=" * 60)
    print("STARTING MACHINE LEARNING MODEL BENCHMARKING")
    print("=" * 60)
    
    results, comparison_df, best_pipe, best_meta = train_and_evaluate_models(df)
    
    print("\n--- MODEL COMPARISON BENCHMARK ---")
    print(comparison_df.to_string(index=False))
    print(f"\nSelected Best Model: {best_meta['best_model_name']}")
    print(f"Test Recall: {best_meta['metrics']['recall']:.4f} | ROC-AUC: {best_meta['metrics']['roc_auc']:.4f} | F1: {best_meta['metrics']['f1_score']:.4f}")
    
    # Save visual
    vis_dir = root / "visualizations"
    vis_file = plot_model_evaluations(results, comparison_df, vis_dir)
    print(f"ROC Curves saved to: {vis_file}")
    
    # Save model bundle
    model_path = save_best_model(best_pipe, best_meta, comparison_df)
    print(f"Best model bundle serialized to: {model_path}")
    
    # Populate SQLite table
    db_file = populate_predictions_in_database(best_pipe, df)
    print(f"Generated predictions for all {len(df)} customers and saved to SQLite at: {db_file}")
    print("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!\n")
    
    return comparison_df, model_path


if __name__ == "__main__":
    run_model_training_pipeline()
