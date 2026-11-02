"""
Top-level entry launcher to run the Customer Churn & Retention Streamlit Dashboard.
Usage:
    python run_dashboard.py
"""

import subprocess
import sys
from pathlib import Path

dashboard_path = Path(__file__).resolve().parent / "customer-churn-retention-analysis" / "dashboard" / "app.py"
if not dashboard_path.exists():
    dashboard_path = Path(__file__).resolve().parent / "dashboard" / "app.py"

print(f"Launching Streamlit Dashboard from: {dashboard_path}")
subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])

