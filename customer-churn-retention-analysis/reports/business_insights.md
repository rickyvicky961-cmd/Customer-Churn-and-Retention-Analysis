# Executive Report: Customer Churn and Retention Analysis

**Prepared For:** Executive Leadership, Growth & Customer Success Teams  
**Date:** November 2026  
**Analyzed Dataset:** IBM Telco Customer Churn (7,043 Accounts, 21 Attributes)  
**Methodology:** Statistical Analysis, Exploratory Data Analysis (EDA), Machine Learning Classification, K-Means Clustering, and SQLite Relational Analytics.

---

## Executive Summary

Customer attrition is one of the single largest headwinds to subscription revenue growth. This analysis provides an end-to-end audit of customer behavior, churn risk predictors, and retention levers across **7,043 customer accounts**.

Key findings indicate an overall customer **churn rate of 26.54%** (1,869 departed customers), leaving an active retention rate of **73.46%**. This level of customer attrition represents an annualized recurring revenue loss of **\$1,669,570.20** (\$139,130.85/month).

Critically, customer churn is not evenly distributed:
1. **Early Tenure Vulnerability:** Customers in their first 12 months exhibit an alarming **47.4% churn rate**.
2. **Contract Disparity:** Month-to-month contracts experience a **42.7% churn rate**, compared to **11.3%** on 1-year and **2.8%** on 2-year agreements.
3. **Fiber Optic & Support Mismatch:** Fiber optic internet subscribers without Tech Support churn at **49.4%**, demonstrating that premium connectivity without proactive technical assistance leads directly to cancellation.
4. **Payment Friction:** Electronic check users experience **45.3% churn**, vs **~15-16%** for automated credit card and bank transfer payers.

By deploying the prioritized **5-Pillar Retention Strategy** developed in this project, the organization can target reducing overall churn from **26.5% to below 18.0%**, safeguarding an estimated **\$550,000 to \$700,000** in annual recurring revenue.

---

## Direct Answers to Core Business Questions

### 1. What percentage of customers are leaving the company?
- **Finding:** **26.54%** of the customer base has departed (1,869 churned customers out of 7,043).
- **Retention Rate:** **73.46%** (5,174 retained accounts).
- **Financial Exposure:** Departed customers generated \$139,130.85 in monthly billings at the time of departure, equating to **\$1,669,570.20 in lost annualized recurring revenue**.

### 2. What are the main reasons for customer churn?
- **Root Cause 1 — Lack of Switching Barriers:** Month-to-month subscriptions allow customers to leave impulsively when competitors run flash discounts.
- **Root Cause 2 — Premium Tech Dissatisfaction:** Fiber optic customers experience high setup expectations. Without bundled Tech Support and Online Security, minor technical disruptions trigger churn.
- **Root Cause 3 — Billing Friction:** Customers using manual electronic checks encounter monthly payment friction and bill-shock reminders.
- **Root Cause 4 — Deficient Onboarding:** High churn within the initial 90 days indicates inadequate onboarding support and delayed time-to-value.

### 3. Which customer segments have the highest churn rate?
- Using K-Means clustering, the customer base was segmented into four distinct behavioral cohorts:
  - **High-Risk High-Spenders:** **44.68% Churn Rate** (2,218 accounts, avg monthly bill \$79.95, avg tenure 17 months). This segment accounts for the highest financial leakage.
  - **New & Vulnerable Customers:** **29.10% Churn Rate** (1,804 accounts, avg tenure 9.4 months, low lifetime spend).
  - **High-Value Loyal Customers:** **15.15% Churn Rate** (1,941 accounts, avg tenure 59.3 months, avg monthly bill \$92.86).
  - **Low-Engagement / Budget Customers:** **5.46% Churn Rate** (1,080 accounts, avg tenure 53.9 months, avg monthly bill \$32.93).

### 4. Does contract type influence customer retention?
- **Yes, decisively.**
  - **Month-to-Month:** **42.71% Churn** (3,875 accounts).
  - **One-Year Contract:** **11.27% Churn** (1,473 accounts).
  - **Two-Year Contract:** **2.83% Churn** (1,695 accounts).
- Moving a subscriber from a month-to-month plan to a 1-year contract reduces their churn probability by **73.6%**, and moving to a 2-year contract reduces it by **93.4%**.

### 5. Does monthly billing influence churn?
- **Yes.** Churned customers pay substantially higher monthly fees:
  - **Retained Customers:** Mean \$61.27 / month (Median: \$64.43).
  - **Churned Customers:** Mean \$74.44 / month (Median: \$79.65).
- Customers paying in the **\$70 – \$100/month** bracket have the highest concentration of churn. Customers paying under \$35/month have a churn rate under 11%.

### 6. Which payment methods are associated with higher churn?
- **Electronic Check:** **45.29% Churn** (2,365 accounts) — 3x higher than automated channels.
- **Mailed Check:** **19.11% Churn** (1,612 accounts).
- **Bank Transfer (Automatic):** **16.71% Churn** (1,544 accounts).
- **Credit Card (Automatic):** **15.24% Churn** (1,522 accounts).
- Automatic payment methods dramatically stabilize subscriber retention by removing manual transaction resistance.

### 7. How does customer tenure affect churn?
- Churn risk follows a steep decay curve over time:
  - **0 – 12 Months:** **47.4% Churn Rate** (1,037 churned out of 2,186).
  - **13 – 24 Months:** **28.7% Churn Rate** (294 churned out of 1,024).
  - **25 – 48 Months:** **18.4% Churn Rate** (293 churned out of 1,594).
  - **49 – 72 Months:** **10.9% Churn Rate** (245 churned out of 2,239).
- Once a customer reaches 24 months, their survival probability increases by nearly 3x.

### 8. Which services are associated with better retention?
- Protective services provide substantial retention stickiness:
  - **Tech Support:** Customers *with* Tech Support churn at **15.17%**, compared to **41.64%** *without* Tech Support.
  - **Online Security:** Customers *with* Online Security churn at **14.61%**, compared to **41.77%** *without*.
  - **Online Backup:** Customers *with* Online Backup churn at **21.53%**, compared to **39.93%** *without*.
  - **Service Breadth:** Subscribers holding 4 or more active services exhibit a churn rate of just **12.4%**.

### 9. Which customers are at high risk of leaving?
- The composite profile of a **Critical Risk Customer (> 70% churn probability)** is:
  1. Month-to-month contract.
  2. Subscribes to Fiber optic internet.
  3. No Tech Support and no Online Security.
  4. Pays via Electronic check with Paperless billing enabled.
  5. Tenure under 12 months.
  6. Monthly charges exceeding \$75.00.

### 10. What strategies can the company implement to reduce churn?
- Implement the comprehensive **5-Pillar Retention Playbook** outlined below, focusing on contract conversion, service bundling, auto-pay migration, and proactive onboarding.

---

## Machine Learning Model Benchmark

Four classification models were trained using stratified 80/20 train-test splits and leak-free Scikit-Learn pipelines:

| Model | Accuracy | Precision | Recall (Priority) | F1-Score | ROC-AUC | Primary Strength |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (Champion)** | **75.59%** | **52.65%** | **79.68%** | **0.6340** | **0.8430** | Superior Recall & Balanced ROC-AUC |
| **Logistic Regression** | 74.24% | 50.97% | 77.54% | 0.6151 | 0.8460 | Excellent Probabilistic Calibration |
| **Gradient Boosting** | 80.48% | 66.45% | 53.48% | 0.5926 | 0.8461 | Highest Accuracy & Precision |
| **Decision Tree** | 74.38% | 51.11% | 79.95% | 0.6236 | 0.8309 | Fast, Interpretable Rules |

> **Selection Rationale:** In customer churn retention, **Recall is paramount**. A False Negative (failing to identify a churning customer) leads to irreversible revenue loss, whereas a False Positive (offering a retention perk to a customer who might have stayed) carries minimal operational cost. The **Random Forest** achieved **79.68% test recall** and **0.843 ROC-AUC**, successfully catching 8 out of every 10 departing accounts.

---

## The 5-Pillar Retention Strategy Playbook

```
+-----------------------------------------------------------------------------------------+
|                                5-PILLAR RETENTION PLAYBOOK                               |
+-----------------------------------------------------------------------------------------+
| 1. Contract Migration    --> 10% annual discount to lock month-to-month subscribers.     |
| 2. Security Bundling     --> Package TechSupport + OnlineSecurity into standard Fiber.   |
| 3. Auto-Pay Modernization--> $5/month billing credit for ACH / Credit Card Auto-Pay.     |
| 4. First-90-Day VIP      --> Automated concierge onboarding at Day-7, Day-30, Day-60.    |
| 5. At-Risk Alert Desk    --> Webhook-triggered proactive CS desk for high-value risk.   |
+-----------------------------------------------------------------------------------------+
```

### Pillar 1: Contract Migration Campaign
- **Target:** 3,875 month-to-month subscribers.
- **Action:** Offer an incentive: "Commit to 12 months, receive a fixed 10% rate discount or 1 free month."
- **Projected Impact:** Migrating 25% of month-to-month subscribers to 1-year agreements saves ~300 accounts annually, protecting **\$270,000+** in revenue.

### Pillar 2: "Complete Home Fiber" Bundle
- **Target:** 3,096 Fiber optic internet users.
- **Action:** Stop selling bare Fiber optic connections. Bundle Tech Support & Online Security into a unified tier.
- **Projected Impact:** Reduces Fiber churn rate from 41.9% to ~22%, cutting annual customer losses by **~400 accounts**.

### Pillar 3: Auto-Pay Adoption Incentive
- **Target:** 2,365 Electronic check users.
- **Action:** Introduce an ongoing \$5/month statement credit or one-time \$25 credit for switching to Credit Card or Bank Auto-Pay.
- **Projected Impact:** Cuts electronic check churn from 45.3% down to < 18%, saving **~350 accounts/year**.

### Pillar 4: First-90-Day Onboarding Concierge
- **Target:** All newly acquired subscribers (tenure 0-6 months).
- **Action:** Establish automated check-in triggers (Day 7 setup check, Day 30 satisfaction call, Day 60 speed optimization).
- **Projected Impact:** Decreases year-1 churn from 47.4% to below 30%, retaining **~200 additional customers** annually.

### Pillar 5: Real-Time At-Risk Account Desk
- **Target:** Customers flagged by ML model with Churn Probability > 70% and monthly bill > \$75.
- **Action:** Priority routing directly to dedicated retention specialists authorized to adjust plans, issue credits, or upgrade equipment.
- **Projected Impact:** Protects high-value subscribers with an estimated immediate recovery rate of 35-40%.

---

## Conclusion & ROI Projections

| Initiative | Implementation Time | Estimated CapEx/OpEx | Projected Annual Revenue Saved |
| :--- | :---: | :---: | :---: |
| Contract Migration Campaign | Weeks 1 – 4 | \$35,000 | \$270,000 |
| Auto-Pay Incentive Drive | Weeks 2 – 6 | \$20,000 | \$185,000 |
| Complete Fiber Bundling | Weeks 4 – 8 | \$45,000 | \$160,000 |
| First-90-Day Concierge | Weeks 6 – 10 | \$30,000 | \$95,000 |
| **Total Program Impact** | **10 Weeks** | **\$130,000** | **\$710,000 / Year** |

**Net Program ROI: ~446% in Year 1.**

