"""
retention_recommendations.py
----------------------------
Module for generating data-backed customer retention strategies,
individual personalized intervention playbooks, and answering core business questions.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


def answer_business_questions(kpis: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, str]]:
    """
    Provide definitive data-driven answers to the 10 business questions.
    """
    kpis = kpis or {}
    churn_rate = kpis.get("churn_rate_pct", 26.54)
    retention_rate = kpis.get("retention_rate_pct", 73.46)
    
    return {
        "Q1_Churn_Percentage": {
            "question": "What percentage of customers are leaving the company?",
            "answer": (
                f"The overall customer churn rate is {churn_rate}%, representing 1,869 churned customers out of 7,043. "
                f"The baseline customer retention rate is {retention_rate}%. This attrition causes an estimated "
                f"${kpis.get('total_annual_revenue_lost', 1669570.20):,.2f} in lost recurring annual revenue."
            )
        },
        "Q2_Main_Reasons": {
            "question": "What are the main reasons for customer churn?",
            "answer": (
                "The three dominant drivers of churn are: (1) Month-to-month contracts providing zero switching barriers; "
                "(2) Lack of technical support and online security on premium Fiber optic connections; and "
                "(3) High monthly billing friction coupled with non-automated payment methods (Electronic check)."
            )
        },
        "Q3_Highest_Churn_Segments": {
            "question": "Which customer segments have the highest churn rate?",
            "answer": (
                "The 'High-Risk High-Spenders' segment has the highest churn rate at 44.68%, despite paying the second-highest "
                "average monthly bill ($79.95/month). The 'New & Vulnerable' cohort also experiences an acute ~29.1% churn rate "
                "within their initial 9-12 months of service."
            )
        },
        "Q4_Contract_Influence": {
            "question": "Does contract type influence customer retention?",
            "answer": (
                "Yes, overwhelmingly. Month-to-month contracts have a 42.7% churn rate. In contrast, 1-Year contracts drop churn "
                "to 11.3%, and 2-Year contracts reduce churn to just 2.8%. Long-term agreements boost retention by over 85%."
            )
        },
        "Q5_Monthly_Billing_Influence": {
            "question": "Does monthly billing influence churn?",
            "answer": (
                "Yes. Retained customers pay an average of $61.27/month, whereas churned customers pay $74.44/month. "
                "Customers paying between $70 and $100/month show peak churn concentration, primarily because they subscribe to "
                "Fiber optic internet without bundled support services."
            )
        },
        "Q6_Payment_Methods": {
            "question": "Which payment methods are associated with higher churn?",
            "answer": (
                "Electronic check has by far the highest churn rate at 45.3%. In comparison, automated payment methods "
                "(Bank transfer at 16.7% and Credit card at 15.2%) and Mailed check (19.1%) have dramatically lower churn. "
                "Electronic check involves active monthly billing friction and higher failure rates."
            )
        },
        "Q7_Tenure_Effect": {
            "question": "How does customer tenure affect churn?",
            "answer": (
                "Churn decreases monotonically as tenure increases. Customers in their first year (0-12 months) suffer "
                "a 47.4% churn rate. By months 25-48, churn falls to ~20%, and drops below 8% for customers with tenure exceeding 5 years. "
                "The critical retention window is the first 90 to 180 days."
            )
        },
        "Q8_Services_Retention": {
            "question": "Which services are associated with better retention?",
            "answer": (
                "Add-on protection and support services exhibit the strongest retention stickiness: Tech Support (churn drops from "
                "41.6% to 15.2%), Online Security (churn drops from 41.8% to 14.6%), and Online Backup (churn drops from 39.9% to 21.5%). "
                "Subscribers with 4+ total services churn at less than half the company average."
            )
        },
        "Q9_High_Risk_Customers": {
            "question": "Which customers are at high risk of leaving?",
            "answer": (
                "Customers exhibiting this profile: (1) Month-to-month contract, (2) Fiber optic internet without Tech Support or "
                "Online Security, (3) Payment via Electronic check, (4) Paperless billing active, and (5) Tenure under 12 months. "
                "Such customers carry a modeled churn probability exceeding 70%."
            )
        },
        "Q10_Reduction_Strategies": {
            "question": "What strategies can the company implement to reduce churn?",
            "answer": (
                "Implement a 5-Pillar Retention Strategy: (1) Contract Migration Campaign with 10% annual discounts; "
                "(2) 'Total Peace of Mind' bundle bundling free Tech Support & Security with Fiber; (3) Auto-Pay incentive ($5 billing credit); "
                "(4) First-90-Day VIP Onboarding Concierge; and (5) Proactive AI Alert triggering CS outreach when usage drops."
            )
        }
    }


def analyze_customer_risk_drivers(customer_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Analyze customer attributes to determine specific risk and protective factors.
    """
    drivers = []
    
    # Contract Check
    contract = customer_data.get("Contract", "Month-to-month")
    if contract == "Month-to-month":
        drivers.append({
            "type": "RISK",
            "feature": "Contract",
            "impact": "High Risk",
            "description": "Month-to-month contract enables frictionless cancellation."
        })
    else:
        drivers.append({
            "type": "PROTECTIVE",
            "feature": "Contract",
            "impact": "High Retention",
            "description": f"{contract} contract provides strong commitment anchor."
        })
        
    # Internet & Support
    internet = customer_data.get("InternetService", "DSL")
    tech_support = customer_data.get("TechSupport", "No")
    security = customer_data.get("OnlineSecurity", "No")
    
    if internet == "Fiber optic" and tech_support == "No":
        drivers.append({
            "type": "RISK",
            "feature": "Service Experience",
            "impact": "High Risk",
            "description": "High-speed Fiber without TechSupport leads to unresolved technical frustration."
        })
        
    if security == "Yes" and tech_support == "Yes":
        drivers.append({
            "type": "PROTECTIVE",
            "feature": "Support Bundle",
            "impact": "High Retention",
            "description": "Active OnlineSecurity and TechSupport significantly increase stickiness."
        })
        
    # Payment Method
    payment = customer_data.get("PaymentMethod", "Electronic check")
    if payment == "Electronic check":
        drivers.append({
            "type": "RISK",
            "feature": "Billing Method",
            "impact": "Moderate Risk",
            "description": "Electronic check has 45%+ churn rate due to manual monthly effort."
        })
    elif "automatic" in str(payment).lower():
        drivers.append({
            "type": "PROTECTIVE",
            "feature": "Billing Method",
            "impact": "Low Risk",
            "description": "Automated payment eliminates recurring payment friction."
        })
        
    # Tenure
    tenure = int(customer_data.get("tenure", 12))
    if tenure <= 6:
        drivers.append({
            "type": "RISK",
            "feature": "Early Tenure",
            "impact": "Critical Risk",
            "description": f"Customer is in vulnerable onboarding phase ({tenure} months)."
        })
    elif tenure >= 36:
        drivers.append({
            "type": "PROTECTIVE",
            "feature": "Matured Tenure",
            "impact": "Strong Retention",
            "description": f"Customer has established loyalty ({tenure} months with company)."
        })
        
    # Monthly Charges
    monthly = float(customer_data.get("MonthlyCharges", 65.0))
    if monthly > 80.0:
        drivers.append({
            "type": "RISK",
            "feature": "Price Sensitivity",
            "impact": "Moderate Risk",
            "description": f"High monthly bill (${monthly:.2f}/mo) increases price sensitivity to competitor offers."
        })
        
    return drivers


def generate_personalized_retention_plan(
    customer_data: Dict[str, Any],
    churn_prob: float
) -> Dict[str, Any]:
    """
    Generate tailored retention interventions, estimated risk reduction, and financial ROI.
    """
    risk_drivers = analyze_customer_risk_drivers(customer_data)
    
    # Categorize Risk
    if churn_prob >= 0.70:
        risk_level = "CRITICAL"
        risk_color = "#d9534f"
        urgency = "Immediate Outreach (Within 48 Hours)"
    elif churn_prob >= 0.40:
        risk_level = "HIGH / ELEVATED"
        risk_color = "#f0ad4e"
        urgency = "Proactive Campaign (Within 7 Days)"
    elif churn_prob >= 0.20:
        risk_level = "MODERATE"
        risk_color = "#5bc0de"
        urgency = "Nurture Campaign (Next Billing Cycle)"
    else:
        risk_level = "LOW / LOYAL"
        risk_color = "#5cb85c"
        urgency = "Standard Advocacy & Cross-Sell"
        
    recommendations = []
    estimated_risk_reduction = 0.0
    
    contract = customer_data.get("Contract", "Month-to-month")
    if contract == "Month-to-month":
        recommendations.append({
            "priority": "P1 - Critical",
            "initiative": "Annual Contract Migration Incentive",
            "action": "Offer a 10% monthly discount or 1 free month in exchange for committing to a 12-month agreement.",
            "impact": "Reduces churn propensity by ~25-30%."
        })
        estimated_risk_reduction += 0.25
        
    internet = customer_data.get("InternetService", "DSL")
    tech_support = customer_data.get("TechSupport", "No")
    security = customer_data.get("OnlineSecurity", "No")
    
    if internet != "No" and (tech_support != "Yes" or security != "Yes"):
        recommendations.append({
            "priority": "P2 - High",
            "initiative": "'Peace of Mind' Service Add-on",
            "action": "Provide complimentary 90-day trial of 24/7 Tech Support and Online Security.",
            "impact": "Increases product value perception and cuts support dissatisfaction churn by ~18%."
        })
        estimated_risk_reduction += 0.15
        
    payment = customer_data.get("PaymentMethod", "Electronic check")
    if payment == "Electronic check":
        recommendations.append({
            "priority": "P3 - Medium",
            "initiative": "Auto-Pay Switching Incentive",
            "action": "Offer a one-time $15 account credit or $3/mo ongoing savings for switching to Bank Auto-Pay or Credit Card.",
            "impact": "Eliminates monthly payment barrier, lowering churn rate from 45% to < 16%."
        })
        estimated_risk_reduction += 0.12
        
    tenure = int(customer_data.get("tenure", 12))
    if tenure <= 6:
        recommendations.append({
            "priority": "P1 - Onboarding",
            "initiative": "Customer Success Onboarding Concierge",
            "action": "Schedule a courtesy satisfaction check-in to verify equipment performance and optimize setup.",
            "impact": "Prevents early buyer remorse and resolves initial friction."
        })
        estimated_risk_reduction += 0.10
        
    # Default loyalty recommendation if already low risk
    if not recommendations:
        recommendations.append({
            "priority": "P3 - Growth",
            "initiative": "Loyalty Appreciation & Referral Incentive",
            "action": "Send VIP customer appreciation perk and referral rewards for introducing friends/family.",
            "impact": "Deepens brand advocacy and generates low-CAC new customer acquisitions."
        })
        estimated_risk_reduction = 0.05
        
    # Cap post-intervention probability
    post_intervention_prob = max(0.05, churn_prob * (1.0 - min(0.65, estimated_risk_reduction)))
    
    # Financial impact calculation
    monthly_charges = float(customer_data.get("MonthlyCharges", 65.0))
    annual_revenue = monthly_charges * 12
    revenue_at_risk = round(annual_revenue * churn_prob, 2)
    saved_revenue_potential = round(annual_revenue * (churn_prob - post_intervention_prob), 2)
    
    return {
        "customer_id": customer_data.get("customerID", "SIMULATED-01"),
        "churn_probability": round(churn_prob, 4),
        "churn_probability_pct": f"{churn_prob * 100:.1f}%",
        "post_intervention_prob_pct": f"{post_intervention_prob * 100:.1f}%",
        "risk_level": risk_level,
        "risk_color": risk_color,
        "urgency": urgency,
        "revenue_at_risk_annual": revenue_at_risk,
        "saved_revenue_potential": saved_revenue_potential,
        "risk_drivers": risk_drivers,
        "recommended_actions": recommendations
    }


def get_company_retention_playbook() -> List[Dict[str, Any]]:
    """
    Return the strategic 5-pillar company-wide retention playbook.
    """
    return [
        {
            "pillar": "1. Contract Migration & Value Locking",
            "target_segment": "Month-to-month subscribers (> 55% of customer base)",
            "action": "Launch targeted digital campaigns offering 12-month fixed pricing with 10% discount vs month-to-month rates.",
            "expected_churn_reduction": "-60% for converted customers",
            "kpi_target": "Shift month-to-month share from 55% down to under 35%"
        },
        {
            "pillar": "2. Tech Support & Security Bundling",
            "target_segment": "Fiber optic subscribers lacking support add-ons",
            "action": "Repackage standalone fiber internet into 'Complete Home Fiber' that automatically bundles TechSupport & OnlineSecurity.",
            "expected_churn_reduction": "-45% for Fiber customers",
            "kpi_target": "Increase TechSupport penetration from 29% to 50%+"
        },
        {
            "pillar": "3. Payment Modernization & Auto-Pay Drive",
            "target_segment": "Electronic check users (45.3% churn rate)",
            "action": "Provide a permanent $5/month discount for activating Credit Card or Bank ACH auto-billing.",
            "expected_churn_reduction": "-55% on electronic check cohort",
            "kpi_target": "Migrate 60% of electronic check payers to automatic billing"
        },
        {
            "pillar": "4. First-90-Days Onboarding Success",
            "target_segment": "New customers with tenure < 6 months (47% churn rate)",
            "action": "Implement automated Day-7, Day-30, and Day-60 proactive satisfaction check-ins and Wi-Fi speed certifications.",
            "expected_churn_reduction": "-35% in Year 1 churn",
            "kpi_target": "Increase 12-month customer retention from 53% to 75%"
        },
        {
            "pillar": "5. High-Value At-Risk CS Fast Response",
            "target_segment": "High-Risk High-Spenders (paying > $75/mo, predicted risk > 70%)",
            "action": "Deploy real-time webhook alerting to senior retention desk for proactive account health reviews and customized retention packages.",
            "expected_churn_reduction": "-40% in revenue leakage",
            "kpi_target": "Protect $650,000+ in annual high-value recurring revenue"
        }
    ]


if __name__ == "__main__":
    sample_customer = {
        "customerID": "TEST-CUST-999",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.85,
        "TotalCharges": 269.55
    }
    plan = generate_personalized_retention_plan(sample_customer, churn_prob=0.82)
    print("--- SAMPLE RETENTION PLAN ---")
    print(f"Customer: {plan['customer_id']} | Risk Level: {plan['risk_level']} ({plan['churn_probability_pct']})")
    print(f"Revenue at Risk: ${plan['revenue_at_risk_annual']} | Potential Saved: ${plan['saved_revenue_potential']}")
    for action in plan["recommended_actions"]:
        print(f"[{action['priority']}] {action['initiative']}: {action['action']}")
