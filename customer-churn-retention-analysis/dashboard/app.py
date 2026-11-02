"""
Customer Churn & Retention Analytics Intelligence Platform
Streamlit Web Application
"""

import os
import sys
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure base directory is in sys.path
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.retention_recommendations import (
    generate_personalized_retention_plan,
    answer_business_questions,
    get_company_retention_playbook
)


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn & Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .metric-card-subtitle {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 2px;
    }
    .risk-badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #FFEDD5;
        color: #9A3412;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
    }
    .risk-badge-low {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Resource Loading & Caching
# ---------------------------------------------------------
@st.cache_data
def load_datasets() -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """Load cleaned dataset, KPIs, and segmentation profile."""
    data_path = _project_root / "data" / "processed" / "cleaned_customer_data.csv"
    if not data_path.exists():
        from src.data_cleaning import run_data_cleaning_pipeline
        df, _ = run_data_cleaning_pipeline()
    else:
        df = pd.read_csv(data_path)
        
    db_path = _project_root / "database" / "customer_database.db"
    conn = sqlite3.connect(str(db_path))
    try:
        kpis_df = pd.read_sql_query("SELECT * FROM churn_kpis LIMIT 1;", conn)
        kpis = kpis_df.to_dict(orient="records")[0] if not kpis_df.empty else {}
        try:
            segments_df = pd.read_sql_query("SELECT * FROM customer_segments;", conn)
        except Exception:
            segments_df = pd.DataFrame()
    except Exception:
        from src.exploratory_analysis import calculate_churn_kpis
        kpis = calculate_churn_kpis(df)
        segments_df = pd.DataFrame()
    finally:
        conn.close()
        
    return df, kpis, segments_df


@st.cache_resource
def load_trained_model():
    """Load serialized ML model bundle."""
    model_path = _project_root / "models" / "churn_model.pkl"
    if not model_path.exists():
        from src.model_training import run_model_training_pipeline
        run_model_training_pipeline()
        
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


# Load data and model
df_cleaned, kpis, segments_summary = load_datasets()
model_bundle = load_trained_model()
pipeline = model_bundle["pipeline"]


# ---------------------------------------------------------
# Sidebar Navigation & Global Info
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=64)
    st.title("Churn Intelligence")
    st.caption("AI-Powered Retention & Customer Analytics")
    st.markdown("---")
    
    app_mode = st.radio(
        "Navigation Menu",
        [
            "Executive Overview & KPIs",
            "Exploratory Data Analysis (EDA)",
            "ML Customer Risk Simulator",
            "High-Risk Customer Directory",
            "Customer Segmentation",
            "Retention Strategy & Answers",
            "SQL Query Console"
        ],
        index=0
    )
    
    st.markdown("---")
    st.subheader("Model Diagnostic")
    st.write(f"**Active Model:** {model_bundle.get('best_model_name', 'Random Forest')}")
    st.write(f"**Recall:** {model_bundle['metrics'].get('recall', 0):.2%}")
    st.write(f"**ROC-AUC:** {model_bundle['metrics'].get('roc_auc', 0):.3f}")
    st.caption("IBM Telco Dataset (7,043 Accounts)")


# ---------------------------------------------------------
# 1. Executive Overview & KPIs
# ---------------------------------------------------------
if app_mode == "Executive Overview & KPIs":
    st.markdown("<div class='main-header'>Executive Overview & Retention Scorecard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>High-level operational health, customer attrition KPIs, and financial exposure.</div>", unsafe_allow_html=True)
    
    # KPI Grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Total Customer Accounts</div>
            <div class="metric-card-value">{kpis.get('total_customers', len(df_cleaned)):,}</div>
            <div class="metric-card-subtitle">Active Base Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        churn_rate = kpis.get('churn_rate_pct', 26.54)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Overall Churn Rate</div>
            <div class="metric-card-value" style="color: #DC2626;">{churn_rate:.1f}%</div>
            <div class="metric-card-subtitle">{kpis.get('churned_customers', 0):,} Customers Departed</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        retention_rate = kpis.get('retention_rate_pct', 73.46)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Customer Retention Rate</div>
            <div class="metric-card-value" style="color: #16A34A;">{retention_rate:.1f}%</div>
            <div class="metric-card-subtitle">{kpis.get('active_customers', 0):,} Retained Accounts</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Annual Revenue Lost</div>
            <div class="metric-card-value" style="color: #B91C1C;">${kpis.get('total_annual_revenue_lost', 0):,.0f}</div>
            <div class="metric-card-subtitle">${kpis.get('total_monthly_revenue_lost', 0):,.0f} / Month at Risk</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Avg Customer Tenure</div>
            <div class="metric-card-value">{kpis.get('avg_customer_tenure_months', 0):.1f} <span style="font-size: 1rem; color:#64748B;">Months</span></div>
            <div class="metric-card-subtitle">Lifetime Duration</div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Avg Monthly Charges</div>
            <div class="metric-card-value">${kpis.get('avg_monthly_charges', 0):.2f}</div>
            <div class="metric-card-subtitle">Portfolio Mean ARPU</div>
        </div>
        """, unsafe_allow_html=True)
    with col7:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Retained ARPU</div>
            <div class="metric-card-value" style="color: #2563EB;">${kpis.get('avg_monthly_revenue_retained', 0):.2f}</div>
            <div class="metric-card-subtitle">Monthly Charge (Retained)</div>
        </div>
        """, unsafe_allow_html=True)
    with col8:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Churned ARPU</div>
            <div class="metric-card-value" style="color: #EA580C;">${kpis.get('avg_monthly_revenue_churned', 0):.2f}</div>
            <div class="metric-card-subtitle">Higher Monthly Bill Exposure</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Overview Visuals
    c_left, c_right = st.columns([1, 1])
    with c_left:
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Retained Customers", "Churned Customers"],
            values=[kpis.get('active_customers', 5174), kpis.get('churned_customers', 1869)],
            hole=.45,
            marker_colors=["#2563EB", "#EF4444"],
            textinfo="label+percent+value",
            textposition="inside"
        )])
        fig_donut.update_layout(
            title="<b>Customer Retention vs Attrition Split</b>",
            margin=dict(l=20, r=20, t=50, b=20),
            height=380
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with c_right:
        # ARPU Comparison
        fig_arpu = go.Figure(data=[
            go.Bar(
                name="Monthly Charges",
                x=["Retained Accounts", "Churned Accounts"],
                y=[kpis.get('avg_monthly_revenue_retained', 61.27), kpis.get('avg_monthly_revenue_churned', 74.44)],
                marker_color=["#2563EB", "#DC2626"],
                text=[f"${kpis.get('avg_monthly_revenue_retained', 61.27):.2f}", f"${kpis.get('avg_monthly_revenue_churned', 74.44):.2f}"],
                textposition="auto"
            )
        ])
        fig_arpu.update_layout(
            title="<b>Monthly Spending Disparity (Retained vs Churned)</b>",
            yaxis_title="Average Monthly Bill ($)",
            margin=dict(l=20, r=20, t=50, b=20),
            height=380
        )
        st.plotly_chart(fig_arpu, use_container_width=True)
        
    st.info("💡 **Key Takeaway:** Churned customers paid **$13.17 more per month** on average than retained customers. This confirms churn is heavily concentrated among high-spending users who do not feel they are receiving commensurate value.")


# ---------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
elif app_mode == "Exploratory Data Analysis (EDA)":
    st.markdown("<div class='main-header'>Exploratory Data Analysis & Attrition Drivers</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Interactive cohort breakdown exploring the root causes of customer churn.</div>", unsafe_allow_html=True)
    
    # Interactive Filters
    with st.expander("🔍 Interactive Data Filters", expanded=True):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            contract_filter = st.multiselect("Contract Type", options=df_cleaned["Contract"].unique(), default=df_cleaned["Contract"].unique())
        with fcol2:
            internet_filter = st.multiselect("Internet Service", options=df_cleaned["InternetService"].unique(), default=df_cleaned["InternetService"].unique())
        with fcol3:
            payment_filter = st.multiselect("Payment Method", options=df_cleaned["PaymentMethod"].unique(), default=df_cleaned["PaymentMethod"].unique())
            
        tenure_range = st.slider("Tenure Range (Months)", 0, 72, (0, 72))
        
    # Apply filters
    filtered_df = df_cleaned[
        (df_cleaned["Contract"].isin(contract_filter)) &
        (df_cleaned["InternetService"].isin(internet_filter)) &
        (df_cleaned["PaymentMethod"].isin(payment_filter)) &
        (df_cleaned["tenure"].between(tenure_range[0], tenure_range[1]))
    ]
    
    st.caption(f"Showing **{len(filtered_df):,}** out of **{len(df_cleaned):,}** accounts based on selected filters.")
    
    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Contracts & Billing", "Services & Support", "Tenure & Demographics"])
    
    with eda_tab1:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            # Contract breakdown
            contract_churn = filtered_df.groupby("Contract", observed=False)["Churn_Numeric"].agg(["count", "mean"]).reset_index()
            contract_churn["mean"] = (contract_churn["mean"] * 100).round(1)
            fig_cont = px.bar(
                contract_churn, x="Contract", y="mean", text="mean",
                color="Contract",
                title="<b>Churn Rate by Contract Commitment (%)</b>",
                labels={"mean": "Churn Rate (%)", "Contract": "Contract Type"}
            )
            fig_cont.update_traces(texttemplate="%{text}%", textposition="outside")
            st.plotly_chart(fig_cont, use_container_width=True)
            st.caption("Month-to-month contracts suffer dramatically higher attrition (~42.7%) than 1-Year (11.3%) and 2-Year (2.8%) agreements.")
            
        with col_c2:
            # Payment Method breakdown
            pay_churn = filtered_df.groupby("PaymentMethod", observed=False)["Churn_Numeric"].agg(["count", "mean"]).reset_index()
            pay_churn["mean"] = (pay_churn["mean"] * 100).round(1)
            fig_pay = px.bar(
                pay_churn, x="PaymentMethod", y="mean", text="mean",
                color="PaymentMethod",
                title="<b>Churn Rate by Payment Method (%)</b>",
                labels={"mean": "Churn Rate (%)", "PaymentMethod": "Payment Channel"}
            )
            fig_pay.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_pay.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig_pay, use_container_width=True)
            st.caption("Electronic check customers experience 45.3% churn due to manual billing friction, compared to ~15-16% on auto-pay.")
            
    with eda_tab2:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            # Tech Support Impact
            ts_churn = filtered_df.groupby(["InternetService", "TechSupport"], observed=False)["Churn_Numeric"].mean().reset_index()
            ts_churn["Churn_Pct"] = (ts_churn["Churn_Numeric"] * 100).round(1)
            fig_ts = px.bar(
                ts_churn, x="InternetService", y="Churn_Pct", color="TechSupport", barmode="group",
                title="<b>Churn Rate by Internet Service & Tech Support Subscription</b>",
                labels={"Churn_Pct": "Churn Rate (%)"}
            )
            st.plotly_chart(fig_ts, use_container_width=True)
            st.caption("Fiber optic customers lacking Tech Support churn at over 49%, whereas those with Tech Support churn at only ~23%.")
            
        with col_s2:
            # Service count boxplot
            fig_serv = px.histogram(
                filtered_df, x="MonthlyCharges", color="Churn",
                marginal="box", nbins=40,
                title="<b>Monthly Billing Distribution by Churn Status</b>",
                color_discrete_map={"No": "#2563EB", "Yes": "#DC2626"}
            )
            st.plotly_chart(fig_serv, use_container_width=True)
            st.caption("Bimodal distribution: Non-churners cluster under $25/mo (basic) and $70-$85/mo. Churners spike strongly between $70 and $105/mo.")
            
    with eda_tab3:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig_ten = px.histogram(
                filtered_df, x="tenure", color="Churn",
                nbins=36, barmode="overlay",
                title="<b>Customer Tenure Distribution (Months)</b>",
                color_discrete_map={"No": "#2563EB", "Yes": "#DC2626"}
            )
            st.plotly_chart(fig_ten, use_container_width=True)
            st.caption("Massive attrition spike in months 1-6. After month 24, attrition slows significantly.")
            
        with col_d2:
            senior_churn = filtered_df.groupby("SeniorCitizen", observed=False)["Churn_Numeric"].agg(["count", "mean"]).reset_index()
            senior_churn["mean"] = (senior_churn["mean"] * 100).round(1)
            senior_churn["SeniorCitizen"] = senior_churn["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior Citizen"})
            fig_sr = px.bar(
                senior_churn, x="SeniorCitizen", y="mean", text="mean", color="SeniorCitizen",
                title="<b>Churn Rate by Senior Citizen Status (%)</b>"
            )
            fig_sr.update_traces(texttemplate="%{text}%", textposition="outside")
            st.plotly_chart(fig_sr, use_container_width=True)
            st.caption("Senior citizens have a 41.7% churn rate, largely driven by higher adoption of paperless billing and lack of hands-on setup support.")


# ---------------------------------------------------------
# 3. ML Customer Risk Simulator
# ---------------------------------------------------------
elif app_mode == "ML Customer Risk Simulator":
    st.markdown("<div class='main-header'>Customer Churn Risk Simulator</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Perform live inference on any customer profile and generate instant retention playbooks.</div>", unsafe_allow_html=True)
    
    with st.form("simulation_form"):
        st.subheader("Customer Demographics & Account Information")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        with col_f2:
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        with col_f3:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
        with col_f4:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            
        st.markdown("---")
        st.subheader("Services & Subscriptions")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"] if phone_service == "Yes" else ["No phone service"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        with col_s2:
            online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"])
        with col_s3:
            device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"])
        with col_s4:
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"])
            streaming_mov = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"])
            
        st.markdown("---")
        st.subheader("Tenure & Financial Billing")
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            tenure = st.slider("Customer Tenure (Months)", 0, 72, 6)
        with col_b2:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=140.0, value=85.50, step=1.0)
        with col_b3:
            suggested_total = round(monthly_charges * max(1, tenure), 2)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(suggested_total), step=10.0)
            
        submit_btn = st.form_submit_button("Predict Customer Churn Risk & Generate Retention Plan", type="primary", use_container_width=True)
        
    if submit_btn:
        customer_dict = {
            "customerID": "SIMULATED-ACCOUNT",
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_prot,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_mov,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        
        # Inference via ML pipeline
        inference_df = pd.DataFrame([customer_dict])
        prob = pipeline.predict_proba(inference_df)[:, 1][0]
        
        plan = generate_personalized_retention_plan(customer_dict, churn_prob=prob)
        
        st.markdown("---")
        st.subheader("Machine Learning Prediction Result")
        
        col_r1, col_r2, col_r3 = st.columns([1, 1, 1])
        with col_r1:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={'suffix': "%"},
                title={'text': "<b>Churn Probability</b>"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1F2937"},
                    'steps': [
                        {'range': [0, 40], 'color': "#86EFAC"},
                        {'range': [40, 70], 'color': "#FDE047"},
                        {'range': [70, 100], 'color': "#FCA5A5"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col_r2:
            st.markdown(f"### Risk Classification")
            if plan["risk_level"] == "CRITICAL":
                st.markdown("<span class='risk-badge-critical'>🚨 CRITICAL ATTRITION RISK</span>", unsafe_allow_html=True)
            elif plan["risk_level"].startswith("HIGH"):
                st.markdown("<span class='risk-badge-high'>⚠️ HIGH ATTRITION RISK</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='risk-badge-low'>✅ HEALTHY ACCOUNT (LOW RISK)</span>", unsafe_allow_html=True)
                
            st.write(f"**Action Urgency:** {plan['urgency']}")
            st.write(f"**Annual Revenue Exposed:** ${plan['revenue_at_risk_annual']:,.2f}")
            st.write(f"**Post-Intervention Target:** {plan['post_intervention_prob_pct']}")
            st.write(f"**Potential Revenue Saved:** ${plan['saved_revenue_potential']:,.2f}")
            
        with col_r3:
            st.markdown("### Primary Risk Drivers")
            for driver in plan["risk_drivers"]:
                icon = "🔴" if driver["type"] == "RISK" else "🟢"
                st.write(f"{icon} **{driver['feature']} ({driver['impact']}):** {driver['description']}")
                
        st.markdown("---")
        st.subheader("🎯 Prescribed Action Plan & Retention Interventions")
        for rec in plan["recommended_actions"]:
            with st.container():
                st.markdown(f"**[{rec['priority']}] {rec['initiative']}**")
                st.markdown(f"- **Recommended Action:** {rec['action']}")
                st.markdown(f"- **Expected Outcome:** {rec['impact']}")
                st.write("")


# ---------------------------------------------------------
# 4. High-Risk Customer Directory
# ---------------------------------------------------------
elif app_mode == "High-Risk Customer Directory":
    st.markdown("<div class='main-header'>High-Risk Customer Directory & Batch Pipeline</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Real-time list of current accounts sorted by predicted attrition risk.</div>", unsafe_allow_html=True)
    
    db_path = _project_root / "database" / "customer_database.db"
    conn = sqlite3.connect(str(db_path))
    try:
        preds_df = pd.read_sql_query("SELECT * FROM churn_predictions;", conn)
    except Exception:
        preds_df = pd.DataFrame()
    finally:
        conn.close()
        
    if preds_df.empty:
        st.warning("Prediction table not found in database. Running batch inference...")
        from src.model_training import populate_predictions_in_database
        populate_predictions_in_database(pipeline, df_cleaned)
        conn = sqlite3.connect(str(db_path))
        preds_df = pd.read_sql_query("SELECT * FROM churn_predictions;", conn)
        conn.close()
        
    col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
    with col_f1:
        risk_filter = st.selectbox("Filter by Risk Tier", ["All Tiers", "High Risk (>= 70%)", "Medium Risk (40-70%)", "Low Risk (< 40%)"])
    with col_f2:
        search_id = st.text_input("Search Customer ID", "")
    with col_f3:
        min_monthly = st.slider("Min Monthly Charges ($)", 18, 120, 18)
        
    # Apply filtering
    df_display = preds_df.copy()
    if "High Risk" in risk_filter:
        df_display = df_display[df_display["RiskTier"] == "High Risk"]
    elif "Medium Risk" in risk_filter:
        df_display = df_display[df_display["RiskTier"] == "Medium Risk"]
    elif "Low Risk" in risk_filter:
        df_display = df_display[df_display["RiskTier"] == "Low Risk"]
        
    if search_id.strip():
        df_display = df_display[df_display["customerID"].str.contains(search_id.strip(), case=False)]
        
    df_display = df_display[df_display["MonthlyCharges"] >= min_monthly]
    df_display = df_display.sort_values(by="ChurnProbability", ascending=False)
    
    st.write(f"Displaying **{len(df_display):,}** accounts matching criteria.")
    st.dataframe(
        df_display[["customerID", "RiskTier", "ChurnProbability", "MonthlyCharges", "tenure", "Contract", "Churn"]].rename(columns={
            "customerID": "Customer ID",
            "RiskTier": "Risk Category",
            "ChurnProbability": "Churn Propensity",
            "MonthlyCharges": "Monthly ($)",
            "tenure": "Tenure (Mo)",
            "Contract": "Agreement",
            "Churn": "Actual Status"
        }),
        use_container_width=True,
        height=400
    )
    
    # CSV Download
    csv_data = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Filtered High-Risk Accounts (CSV)",
        data=csv_data,
        file_name="high_risk_retention_targets.csv",
        mime="text/csv",
        type="primary"
    )


# ---------------------------------------------------------
# 5. Customer Segmentation
# ---------------------------------------------------------
elif app_mode == "Customer Segmentation":
    st.markdown("<div class='main-header'>Customer Behavioral Segmentation</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Unsupervised K-Means clustering identifying distinct subscriber archetypes.</div>", unsafe_allow_html=True)
    
    if segments_summary.empty:
        from src.customer_segmentation import perform_customer_clustering
        _, segments_summary, _, _ = perform_customer_clustering(df_cleaned, k=4)
        
    st.subheader("Segment Architecture & Performance Profile")
    st.dataframe(
        segments_summary.rename(columns={
            "Segment_Name": "Segment Archetype",
            "Customer_Count": "Account Volume",
            "Pct_of_Total_Customers": "Share (%)",
            "Avg_Tenure_Months": "Avg Tenure (Mo)",
            "Avg_Monthly_Charges": "Avg Monthly ($)",
            "Avg_Total_Charges": "Avg Lifetime ($)",
            "Avg_Services": "Avg Active Services",
            "Churn_Rate_Pct": "Churn Rate (%)",
            "Total_Monthly_Revenue": "Monthly Revenue ($)"
        }),
        use_container_width=True
    )
    
    col_seg1, col_seg2 = st.columns(2)
    with col_seg1:
        # Scatter 2D
        fig_scatter = px.scatter(
            df_cleaned, x="tenure", y="MonthlyCharges", color="Segment_Name",
            opacity=0.6,
            title="<b>Tenure vs Monthly Charges Across Segments</b>",
            labels={"tenure": "Tenure (Months)", "MonthlyCharges": "Monthly Charges ($)", "Segment_Name": "Segment"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_seg2:
        # Churn rate bar
        fig_seg_churn = px.bar(
            segments_summary, x="Segment_Name", y="Churn_Rate_Pct", text="Churn_Rate_Pct",
            color="Segment_Name",
            title="<b>Attrition Vulnerability by Segment (%)</b>",
            labels={"Churn_Rate_Pct": "Churn Rate (%)", "Segment_Name": "Segment"}
        )
        fig_seg_churn.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig_seg_churn, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Segment Playbooks")
    
    scol1, scol2 = st.columns(2)
    with scol1:
        st.info("""
        **1. High-Risk High-Spenders (44.7% Churn)**
        - **Profile:** Paying ~$80/mo, moderate tenure (17 mo), high fiber usage without bundled support.
        - **Playbook:** Proactive outreach offering complimentary TechSupport & OnlineSecurity plus 10% discount for 1-year contract migration.
        """)
        st.success("""
        **2. High-Value Loyal (15.2% Churn)**
        - **Profile:** Long tenure (59 mo), high monthly charges ($93/mo), subscribing to 5.8 services.
        - **Playbook:** Loyalty rewards, annual VIP status upgrades, exclusive priority customer service channel.
        """)
    with scol2:
        st.warning("""
        **3. New & Vulnerable (29.1% Churn)**
        - **Profile:** New accounts (avg tenure 9.4 mo), month-to-month contracts, high onboarding friction.
        - **Playbook:** First-90-Day customer success cadence, setup check-ins, automated welcome promotions.
        """)
        st.info("""
        **4. Low-Engagement / Budget (5.5% Churn)**
        - **Profile:** Basic phone or DSL users ($33/mo), highly loyal (54 mo tenure).
        - **Playbook:** Gentle cross-sell of affordable digital add-ons without disrupting pricing satisfaction.
        """)


# ---------------------------------------------------------
# 6. Retention Strategy & Answers
# ---------------------------------------------------------
elif app_mode == "Retention Strategy & Answers":
    st.markdown("<div class='main-header'>Executive Retention Strategy & Business Answers</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Definitive data-backed answers to the 10 business questions and 5-pillar strategic roadmap.</div>", unsafe_allow_html=True)
    
    strat_tab1, strat_tab2 = st.tabs(["Answers to Business Questions", "5-Pillar Retention Roadmap"])
    
    with strat_tab1:
        qa_data = answer_business_questions(kpis)
        for key, item in qa_data.items():
            with st.expander(f"📌 {item['question']}", expanded=True if "Q1" in key or "Q2" in key or "Q10" in key else False):
                st.write(item["answer"])
                
    with strat_tab2:
        playbook = get_company_retention_playbook()
        for pillar in playbook:
            st.markdown(f"### {pillar['pillar']}")
            st.markdown(f"- **Target Population:** {pillar['target_segment']}")
            st.markdown(f"- **Strategic Intervention:** {pillar['action']}")
            st.markdown(f"- **Expected Attrition Impact:** `{pillar['expected_churn_reduction']}`")
            st.markdown(f"- **Target KPI:** `{pillar['kpi_target']}`")
            st.write("")


# ---------------------------------------------------------
# 7. SQL Query Console
# ---------------------------------------------------------
elif app_mode == "SQL Query Console":
    st.markdown("<div class='main-header'>Interactive SQL Analytics Console</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Execute real-time SQL queries against SQLite database: <code>customer_database.db</code></div>", unsafe_allow_html=True)
    
    db_path = _project_root / "database" / "customer_database.db"
    
    preset_queries = {
        "Top 10 High-Risk Customers by Monthly Charges": (
            "SELECT customerID, RiskTier, ChurnProbability, MonthlyCharges, tenure, Contract, Churn "
            "FROM churn_predictions "
            "WHERE RiskTier = 'High Risk' "
            "ORDER BY MonthlyCharges DESC LIMIT 10;"
        ),
        "Churn Rate & Revenue Loss by Contract Type": (
            "SELECT Contract, COUNT(*) as Total_Customers, "
            "SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as Churned_Count, "
            "ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as Churn_Rate_Pct, "
            "ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) as Monthly_Revenue_Lost "
            "FROM customers_cleaned "
            "GROUP BY Contract "
            "ORDER BY Churn_Rate_Pct DESC;"
        ),
        "Payment Method Churn Comparison": (
            "SELECT PaymentMethod, COUNT(*) as Customers, "
            "ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as Churn_Rate_Pct, "
            "ROUND(AVG(MonthlyCharges), 2) as Avg_Monthly_Bill "
            "FROM customers_cleaned "
            "GROUP BY PaymentMethod "
            "ORDER BY Churn_Rate_Pct DESC;"
        ),
        "Tech Support Impact on Fiber Optic Customers": (
            "SELECT InternetService, TechSupport, COUNT(*) as Customer_Volume, "
            "ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as Churn_Rate_Pct "
            "FROM customers_cleaned "
            "WHERE InternetService = 'Fiber optic' "
            "GROUP BY TechSupport;"
        )
    }
    
    preset_choice = st.selectbox("Select Pre-built SQL Query", list(preset_queries.keys()))
    sql_text = st.text_area("SQL Query", value=preset_queries[preset_choice], height=120)
    
    if st.button("Run SQL Query", type="primary"):
        conn = sqlite3.connect(str(db_path))
        try:
            query_res = pd.read_sql_query(sql_text, conn)
            st.success(f"Query returned {len(query_res)} rows.")
            st.dataframe(query_res, use_container_width=True)
            
            # Export
            csv_bytes = query_res.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download SQL Query Results (CSV)",
                data=csv_bytes,
                file_name="sql_query_result.csv",
                mime="text/csv"
            )
        except Exception as err:
            st.error(f"SQL Execution Error: {err}")
        finally:
            conn.close()
            
    st.markdown("---")
    st.caption("Available Tables in Database: `customers_raw`, `customers_cleaned`, `churn_predictions`, `customer_segments`, `churn_kpis`")

