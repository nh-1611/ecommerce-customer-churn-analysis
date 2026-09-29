# E-Commerce Customer Churn Analysis & Predictive Modeling

Wersja polska: [README_PL.md](README_PL.md)

## Executive Summary

This project analyzes the customer churn rate in an e-commerce platform and builds a machine learning model to predict churn risk. The business goal is to identify key drivers of customer attrition and enable the retention team to intervene proactively.

- Total analyzed customers: 3,270 (after cleaning)
- Overall Churn Rate: 16.3%
- Selected Predictive Model: XGBoost Classifier (ROC-AUC: 0.9636, Recall: 0.89)

---

## Data Dictionary

The dataset contains information regarding customer behavior and demographics on the e-commerce platform:

- `Tenure`: Tenure of a customer in the company in months (numeric).
- `WarehouseToHome`: Distance between the warehouse to the customer's home in km (numeric).
- `NumberOfDeviceRegistered`: Total number of devices registered to a particular customer (numeric).
- `PreferedOrderCat`: Preferred order category of a customer in the last month (categorical).
- `SatisfactionScore`: Satisfactory score of a customer on service rated 1–5 (numeric).
- `MaritalStatus`: Marital status of a customer (categorical).
- `NumberOfAddress`: Total number of addresses added for a particular customer (numeric).
- `Complain`: Whether any complaint has been raised in the last month (binary: 0 = No, 1 = Yes).
- `DaySinceLastOrder`: Days since last order by customer (numeric).
- `CashbackAmount`: Average cashback in the last month (numeric).
- `Churn`: Churn flag / target variable (binary: 0 = Retained, 1 = Churned).

---

## Project Architecture

1. Data Cleaning & Preprocessing (Python)
2. Exploratory Data Analysis (EDA) & Correlation Analysis (Python)
3. Interactive Executive Dashboard Creation (Power BI)
4. Feature Engineering & Machine Learning Data Preparation (Python)
5. Model Training & Evaluation – Random Forest & XGBoost (Python)
6. Business Insights & Recommendations

---

## Libraries & Tools Used

* **Data Manipulation & Analysis:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning & Preprocessing:** `scikit-learn` (`StandardScaler`, `train_test_split`, `RandomForestClassifier`, `metrics`)
* **Advanced Modeling:** `xgboost` (`XGBClassifier`)
* **Business Intelligence & Reporting:** Power BI Desktop

---

## Step 1: Data Cleaning & Preprocessing (Python)

The following preprocessing steps were executed in Python:

- Missing Value Imputation: Missing numerical values in `Tenure`, `WarehouseToHome`, and `DaySinceLastOrder` were imputed using the median calculated for each respective variable.
- Deduplication: Identified and dropped duplicate records.
- Type Conversion: Standardized and corrected data types for numerical features.
- Target Imbalance Verification: Checked the class distribution (16.3% Churn vs. 83.7% Retained).

---

## Step 2: Exploratory Data Analysis (EDA) & Correlations

Linear relationships between features and the target variable `Churn` were evaluated using Spearman's Rank Correlation:

- `Tenure` (-0.39): Strongest negative correlation. Longer tenure significantly reduces churn risk.
- `Complain` (+0.26): Strongest positive predictor. Filing a complaint substantially increases churn probability.
- `CashbackAmount` (-0.17) and `DaysSinceLastOrder` (-0.17): Higher cashback rewards and more recent orders moderately safeguard against churn.

---

## Step 3: Executive Power BI Dashboard

An operational and analytical dashboard was designed in Power BI, structured into three primary zones:

1. **KPI Scorecard**: Total Customers (3,270), Churn Rate (16.3%), Complaint Rate (28.2%), Average Tenure (10 months).
2. **Filters Panel**: Slicers for Marital Status, Tenure Group, and Preferred Order Category.
3. **Deep-Dive Analysis**:
   - Impact of Complaints: Churn rate reaches 31.8% for customers who submitted complaints vs. 10.3% for those without complaints.
   - Impact of Tenure: 31.2% churn occurs within the initial 0–6 month window. For customers with over 2 years of tenure, the churn rate drops to 0.0%.
   - Satisfaction vs. Complaint Paradox: High satisfaction scores (5/5) do not protect against churn if accompanied by an active complaint (churn reaches 36.5% in this subgroup).
   - Registered Devices: Churn increases from 7.4% for customers with 2 registered devices to 33.7% for those with 6 registered devices.

---

## Step 4: Feature Engineering & ML Preparation

The following preprocessing steps were completed in Python prior to model training:

1. Categorical Encoding: Applied One-Hot Encoding (`pd.get_dummies(drop_first=True)`).
2. Train-Test Split: Executed an 80/20 stratified split (`stratify=y`) to maintain the class distribution ratio.
3. Feature Scaling: Used `StandardScaler` to normalize continuous numerical variables.

---

## Step 5: Machine Learning Model Performance

Tree-based ensemble algorithms were selected for binary classification. Class imbalance was addressed using `class_weight='balanced'` for Random Forest and `scale_pos_weight` for XGBoost.

Performance on the holdout test set (N = 654):

| Metric | Random Forest | XGBoost (Selected Model) |
| :--- | :---: | :---: |
| ROC-AUC Score | 0.9552 | 0.9636 |
| Recall (Churn = 1) | 0.83 (83%) | 0.89 (89%) |
| Precision (Churn = 1) | 0.70 (70%) | 0.71 (71%) |
| F1-Score (Churn = 1) | 0.76 | 0.79 |
| Accuracy | 0.91 | 0.92 |

Confusion Matrix Breakdown for XGBoost:
- True Negative (Retained correctly): 508
- True Positive (Churn detected correctly): 95 out of 107 customers (89% capture rate)
- False Positive (False alarm): 39
- False Negative (Missed churners): 12

---

## Step 6: Feature Importance

A feature importance comparison highlighted key decision rules used by each algorithm:

- Random Forest: Prioritized continuous variables: `Tenure` (~28.4%), `CashbackAmount` (~16.1%), and `WarehouseToHome` (~10.0%).
- XGBoost: Identified `Tenure` (~22.8%) and `Complain` (~13.2%) as primary drivers of gain, perfectly matching findings from EDA and Power BI. High-value product categories (Laptop & Accessory ~7.8%) also emerged as important factors.

---

## Key Insights

1. **Early Tenure Churn Risk:**
   - 31.2% of customers churn within their first 0–6 months on the platform.
   - Beyond the 2-year mark (24+ months), the churn rate drops close to 0.0%, indicating that customer loyalty increases significantly over time.

2. **Complaints as the Primary Trigger:**
   - Customers who submitted a complaint churn over 3 times more often (31.8%) compared to non-complainers (10.3%).
   - Complaint flag holds the strongest positive correlation with `Churn` (+0.26).

3. **The Satisfaction Paradox:**
   - High self-reported satisfaction (5/5) does not protect a customer from leaving if they experienced an unresolved issue. In this specific segment, churn peaks at 36.5%.
   - General satisfaction reflects overall perception, but active operational friction (complaints) outweighs positive opinions.

4. **Registered Devices Exposure:**
   - As the number of registered devices increases, churn escalates from 7.4% (2 devices) to 33.7% (6 devices).
   - This may indicate account sharing or user experience (UX) consistency issues across multiple devices.

5. **Retention Incentive Buffer:**
   - Higher cashback amounts and shorter periods since the last order show negative correlation with churn (~ -0.17). Rewarding activity effectively retains users within the ecosystem.

---

## Strategic Business Recommendations

Based on EDA findings and XGBoost model outputs, three core retention pillars are established:

#### 1. New Customer Onboarding & Protection (0–6 Months Window)
* **Problem:** Churn peaks at **31.2%** during the initial 6 months before dropping sharply.
* **Action:** 
  * Implement an onboarding communication sequence highlighting key platform benefits.
  * Issue automated cashback incentives during months 2 and 4 to build purchasing habits.

#### 2. Complaint Handling & SLA Acceleration
* **Problem:** Complaints are the single strongest trigger for attrition (31.8% churn), driving the satisfaction paradox where churn reaches **36.5%**.
* **Action:**
  * Flag support tickets from new accounts (tenure < 6 months) for priority response.
  * Reduce SLA response time to 12–24 hours and automate goodwill vouchers or cashback rewards upon resolution.

#### 3. CRM Integration & Automated Churn Scoring
* **Problem:** Reactive outreach after a customer stops ordering is often too late.
* **Action:**
  * Integrate the trained XGBoost model into the CRM to score active accounts weekly.
  * **Automated Retention Workflows:**
    * **High Risk (Probability > 70%):** Flag for direct outbound contact by retention agents with custom offers.
    * **Medium Risk (Probability 40%–70%):** Trigger automated marketing emails with personalized coupons or bonus cashback on preferred categories.
