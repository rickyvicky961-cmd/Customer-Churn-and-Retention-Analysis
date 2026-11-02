"""
customer_segmentation.py
------------------------
Module for behavioral customer segmentation using K-Means clustering.
Identifies:
1. High-Value Loyal Customers
2. New & Vulnerable Customers
3. High-Risk High-Spenders
4. Low-Engagement / Budget Customers
Saves cluster profiles and visualization artifacts.
"""

import os
import sys
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
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.data_cleaning import get_project_root
from src.feature_engineering import CustomerFeatureEngineer


def perform_customer_clustering(
    df: pd.DataFrame,
    k: int = 4,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, KMeans, StandardScaler]:
    """
    Cluster customers based on behavioral, financial, and engagement metrics:
    - tenure (customer relationship length)
    - MonthlyCharges (monthly commitment)
    - TotalCharges (lifetime spend)
    - TotalServices (engagement breadth)
    
    Returns:
        df_segmented: DataFrame with Cluster and Segment_Name
        profile_summary: Aggregated stats per segment
        kmeans: Fitted KMeans model
        scaler: Fitted StandardScaler
    """
    df_copy = df.copy()
    
    # Feature engineering for services if not already present
    if "TotalServices" not in df_copy.columns:
        fe = CustomerFeatureEngineer()
        df_copy = fe.transform(df_copy)
        
    cluster_features = ["tenure", "MonthlyCharges", "TotalCharges", "TotalServices"]
    X = df_copy[cluster_features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_copy["Cluster"] = cluster_labels
    
    # Compute cluster centers in original scale for intelligent naming
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers_original, columns=cluster_features)
    
    # Calculate churn rate per cluster to accurately map business archetypes
    churn_rates = df_copy.groupby("Cluster")["Churn_Numeric"].mean()
    centers_df["ChurnRate"] = churn_rates.values
    
    # Identify archetypes based on metrics:
    # Segment 1: High-Value Loyal (Highest tenure + highest TotalCharges)
    # Segment 2: New & Vulnerable (Lowest tenure + low TotalCharges + high churn rate)
    # Segment 3: High-Risk High-Spenders (High MonthlyCharges + high churn rate)
    # Segment 4: Low-Engagement / Budget (Lowest MonthlyCharges)
    
    segment_mapping = {}
    assigned = set()
    
    # 1. High-Value Loyal: max TotalCharges
    loyal_idx = int(centers_df["TotalCharges"].idxmax())
    segment_mapping[loyal_idx] = "High-Value Loyal"
    assigned.add(loyal_idx)
    
    # 2. Low-Engagement / Budget: min MonthlyCharges among unassigned
    unassigned = [i for i in range(k) if i not in assigned]
    budget_idx = min(unassigned, key=lambda i: centers_df.loc[i, "MonthlyCharges"])
    segment_mapping[budget_idx] = "Low-Engagement / Budget"
    assigned.add(budget_idx)
    
    # 3. New & Vulnerable: lowest tenure among remaining
    unassigned = [i for i in range(k) if i not in assigned]
    new_idx = min(unassigned, key=lambda i: centers_df.loc[i, "tenure"])
    segment_mapping[new_idx] = "New & Vulnerable"
    assigned.add(new_idx)
    
    # 4. Remaining: High-Risk High-Spenders
    unassigned = [i for i in range(k) if i not in assigned]
    if unassigned:
        segment_mapping[unassigned[0]] = "High-Risk High-Spenders"
        
    df_copy["Segment_Name"] = df_copy["Cluster"].map(segment_mapping)
    
    # Profile summary
    profile_summary = df_copy.groupby("Segment_Name").agg(
        Customer_Count=("customerID", "count"),
        Avg_Tenure_Months=("tenure", "mean"),
        Avg_Monthly_Charges=("MonthlyCharges", "mean"),
        Avg_Total_Charges=("TotalCharges", "mean"),
        Avg_Services=("TotalServices", "mean"),
        Churn_Rate_Pct=("Churn_Numeric", lambda x: round(x.mean() * 100, 2)),
        Total_Monthly_Revenue=("MonthlyCharges", "sum")
    ).reset_index()
    
    profile_summary["Pct_of_Total_Customers"] = round(
        profile_summary["Customer_Count"] / len(df_copy) * 100, 2
    )
    
    # Round numerical metrics
    profile_summary["Avg_Tenure_Months"] = profile_summary["Avg_Tenure_Months"].round(1)
    profile_summary["Avg_Monthly_Charges"] = profile_summary["Avg_Monthly_Charges"].round(2)
    profile_summary["Avg_Total_Charges"] = profile_summary["Avg_Total_Charges"].round(2)
    profile_summary["Avg_Services"] = profile_summary["Avg_Services"].round(1)
    profile_summary["Total_Monthly_Revenue"] = profile_summary["Total_Monthly_Revenue"].round(2)
    
    return df_copy, profile_summary, kmeans, scaler


def plot_customer_segments(df_segmented: pd.DataFrame, output_dir: Path) -> str:
    """
    Generate customer_segments.png showing 2D scatter of Tenure vs Monthly Charges,
    annotated with segment clusters and density contours.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    palette = {
        "High-Value Loyal": "#2ca02c",
        "New & Vulnerable": "#d62728",
        "High-Risk High-Spenders": "#ff7f0e",
        "Low-Engagement / Budget": "#1f77b4"
    }
    
    # 1. Scatter Plot
    sns.scatterplot(
        data=df_segmented,
        x="tenure",
        y="MonthlyCharges",
        hue="Segment_Name",
        palette=palette,
        alpha=0.65,
        s=40,
        ax=axes[0]
    )
    axes[0].set_title("Customer Segments: Tenure vs Monthly Charges", fontweight="bold", pad=12)
    axes[0].set_xlabel("Tenure (Months)")
    axes[0].set_ylabel("Monthly Charges ($)")
    axes[0].legend(title="Segment Archetype", loc="upper right")
    
    # 2. Churn Rate & Customer Volume by Segment
    seg_stats = df_segmented.groupby("Segment_Name").agg(
        Count=("customerID", "count"),
        ChurnRate=("Churn_Numeric", lambda x: x.mean() * 100)
    ).reindex(palette.keys()).dropna()
    
    bar_colors = [palette.get(seg, "#333333") for seg in seg_stats.index]
    bars = axes[1].bar(seg_stats.index, seg_stats["ChurnRate"], color=bar_colors, edgecolor="black", width=0.5)
    axes[1].set_title("Churn Rate by Customer Segment (%)", fontweight="bold", pad=12)
    axes[1].set_ylabel("Segment Churn Rate (%)")
    axes[1].set_xlabel("Customer Segment")
    axes[1].set_xticks(range(len(seg_stats.index)))
    axes[1].set_xticklabels(seg_stats.index, rotation=20, ha="right")
    axes[1].set_ylim(0, max(seg_stats["ChurnRate"]) * 1.25)
    
    for bar, count in zip(bars, seg_stats["Count"]):
        h = bar.get_height()
        axes[1].annotate(f"{h:.1f}%\n(N={count:,})", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
        
    plt.suptitle("Customer Behavioral Segmentation & Churn Vulnerability", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    out_file = output_dir / "customer_segments.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out_file)


def run_customer_segmentation_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute customer segmentation, save segment tables to database and processed data CSV.
    """
    root = get_project_root()
    processed_csv = root / "data" / "processed" / "cleaned_customer_data.csv"
    if not processed_csv.exists():
        from src.data_cleaning import run_data_cleaning_pipeline
        df, _ = run_data_cleaning_pipeline()
    else:
        df = pd.read_csv(processed_csv)
        
    print("=" * 60)
    print("EXECUTING CUSTOMER BEHAVIORAL SEGMENTATION (K-MEANS)")
    print("=" * 60)
    
    df_segmented, profile_summary, _, _ = perform_customer_clustering(df, k=4)
    
    print("\n--- SEGMENTATION PROFILES ---")
    print(profile_summary.to_string(index=False))
    
    # Overwrite processed data CSV with segment tags
    df_segmented.to_csv(processed_csv, index=False)
    print(f"Updated {processed_csv} with 'Cluster' and 'Segment_Name' columns.")
    
    # Save visual
    vis_dir = root / "visualizations"
    vis_file = plot_customer_segments(df_segmented, vis_dir)
    print(f"Customer segmentation chart saved to: {vis_file}")
    
    # Save to SQLite
    db_path = root / "database" / "customer_database.db"
    conn = sqlite3.connect(str(db_path))
    try:
        profile_summary.to_sql("customer_segments", conn, if_exists="replace", index=False)
        # Update customers_cleaned table in SQLite with segment data
        df_segmented.to_sql("customers_cleaned", conn, if_exists="replace", index=False)
    finally:
        conn.close()
        
    print(f"Persisted segments table to SQLite database at: {db_path}")
    print("CUSTOMER SEGMENTATION COMPLETED SUCCESSFULLY!\n")
    
    return df_segmented, profile_summary


if __name__ == "__main__":
    run_customer_segmentation_pipeline()
