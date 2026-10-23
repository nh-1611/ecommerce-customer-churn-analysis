# E-Commerce Customer Churn Analysis & Predictive Modeling

Polish version: [README_PL.md](README_PL.md)

*E-commerce Customer Churn*: https://www.kaggle.com/datasets/samuelsemaya/e-commerce-customer-churn/data 

## About the project 

The project analyses the historical user base of an e-commerce service, with particular focus on the churn rate, and builds a machine learning model that predicts the churn risk of new customers. To reflect production conditions, the initial analysis and the selection of the model were carried out exclusively on the previously separated training data. For the model, this was made possible by using cross-validation. The test data is treated as a new customer base, for which a churn risk score is calculated.
The business goal is to identify the key factors influencing a customer's decision to leave, to indicate the group of at-risk users and to enable the retention team to intervene early.

---

## At a glance

**Goal:** identify the drivers of customer churn in e-commerce and flag at-risk customers in a new customer base, so that the retention team can intervene early.

**Data:** public Kaggle dataset, 3,941 observations (after cleaning: 2,712 in the training set, 768 in the test set). Churn rate in the historical data: **16.3%**. The test set serves as a simulation of a new customer base with unknown decisions.

**Approach:** statistical analysis (hypothesis tests with effect sizes) → 5-fold stratified cross-validation inside a pipeline (SMOTE and scaling within folds) → tuned Random Forest → interpretation (SHAP, feature importance, logistic regression as a baseline) → predictions for the new base → two Power BI dashboards.

**Model:** Random Forest (chosen among logistic regression, Random Forest and XGBoost).

| Metric (cross-validation, training set) | Value |
| :--- | :---: |
| ROC-AUC | 0.93 |
| Recall (churn = 1) | 0.73 |
| Precision (churn = 1) | 0.68 |
| F1 (churn = 1) | 0.71 |

Decision threshold: **0.35**

**Predictions for the new base:** **165 of 768 customers (21.5%)** above the risk threshold, including 75 in the critical group (75-100%).

**Key findings:**
- **Tenure is the main driver:** 31.5% of customers churn within the first 0-6 months, and practically none after 2 years (r_rb = -0.60, large effect; the strongest test result).
- **Complaints:** churn of 31.0% for customers with a complaint vs 10.5% without.
- **Registered devices:** churn rises from 6.4% (2 devices) to 35.2% (6 devices).
- **Features according to SHAP:** tenure, marital status (single), preferred category (mobile phones), cashback.
- **Surprising results:** higher satisfaction and fewer days since the last order are associated with more churn, which probably reflects characteristics of the dataset.

**Repository contents:**
- `ecommerce_churn_analysis.ipynb`: cleaning, EDA, hypothesis tests
- `ecommerce_churn_model.ipynb`: modelling, interpretation, prediction
- `ecommerce_churn_analysis_dashboard_pl.pbix`: analytical dashboard in Power BI
- `ecommerce_churn_prediction_pl.pbix`: prediction dashboard in Power BI
- `requirements.txt`: libraries

How to run: pip install -r `requirements.txt`, then run the notebooks in this order: `ecommerce_churn_analysis.ipynb` → `ecommerce_churn_model.ipynb`. 

---

## Variables

3,941 observations before cleaning

The dataset contains information about the behaviour and profile of customers of an e-commerce platform:

- `Tenure`: Time the customer has been using the company's services, in months (numerical).
- `WarehouseToHome`: Distance between the warehouse and the customer's home in km (numerical).
- `NumberOfDeviceRegistered`: Total number of devices registered by the customer (numerical).
- `PreferedOrderCat`: Customer's preferred order category in the last month (categorical).
- `SatisfactionScore`: Customer's satisfaction rating of the service on a scale of 1-5 (numerical).
- `MaritalStatus`: Customer's marital status (categorical).
- `NumberOfAddress`: Total number of addresses added by the customer (numerical).
- `Complain`: Whether a complaint was submitted in the last month (binary: 0 = no, 1 = yes).
- `DaySinceLastOrder`: Number of days since the customer's last order (numerical).
- `CashbackAmount`: Average cashback amount granted in the last month (numerical).
- `Churn`: Target variable indicating customer churn (binary: 0 = stayed, 1 = left).

---

## Project architecture

1. Data preparation, initial split and cleaning (Python)
2. Exploratory data analysis (EDA), normality testing, correlation analysis, multicollinearity analysis, hypothesis testing (Python)
3. Interactive dashboard for the EDA (Power BI)
4. Feature engineering and data preparation for ML models (Python)
5. Training and evaluation of logistic regression, Random Forest and XGBoost models (Python)
6. Interpretation of feature importance (Python)
7. Churn risk prediction for the new customer base (Python)
8. Interactive dashboard for the predictions (Power BI)
9. Findings, business recommendations, limitations and next steps

---

## Tools used

* **Data analysis and processing:** `pandas`, `numpy`, `scipy` (`stats`), `statsmodels`
  
* **Data visualisation:** `matplotlib`, `seaborn`

* **Machine learning models**
* `scikit-learn` - data splitting (`train_test_split`), scaling (`StandardScaler`), hyperparameter optimisation (`GridSearchCV`), cross-validation (`StratifiedKFold`, `cross_validate`) and model evaluation (`classification_report`, `confusion_matrix`, `roc_auc_score`), logistic regression (`LogisticRegression`), random forest (`RandomForestClassifier`)
* `imblearn` - handling class imbalance (`SMOTE`), building processing pipelines (`Pipeline`)
* `xgboost` (`XGBClassifier`)
* `shap` - model interpretation
  
* **Business Intelligence / Reporting:** Power BI Desktop

---


## Step 1: Split into training and test data 

The data was split into training (3,152) and test (789) sets **BEFORE the analysis**, to reflect production conditions. The main purpose is to prevent **data leakage**. Splitting the raw data before imputing missing values, scaling or encoding guarantees that statistics (e.g. medians) are computed exclusively on the training set.
An **80/20 split with preserved class proportions** was used. Stratification keeps identical class proportions (churn vs no churn) in both sets, eliminating the risk of random sample skew.

The training data is treated as the historical customer base, while the test data is meant to reflect a new customer base, for which the model calculates the risk score of deciding to leave the service.

## Step 2: Data cleaning and preparation 

The following operations were carried out in the data preparation stage:

- **Imputation of missing values:** missing numerical values in the training and test sets, in the columns `Tenure`, `WarehouseToHome` and `DaySinceLastOrder`, were filled with the median computed for each of these variables on the training set.
- **Removing duplicates:** repeated records in the dataset were identified and removed.
- **Type conversion:** data types of numerical columns were corrected.
- **Outlier capping:** lower and upper bounds were determined for each variable using the interquartile range (IQR), and outliers were then capped to these bounds. The *capping* method limits the influence of extreme values while preserving information about the data. For the test set, the bounds computed on the training set were used.
- **Value mapping:** incorrectly labelled records were unified.

After cleaning, the training set contains 2,712 observations and the test set 768.

---

## Step 3: Exploratory Data Analysis (EDA)

Descriptive statistics were analysed for the training data.

> * The cleaned historical customer base has **2,712 records.**
> * The customer base is imbalanced. About **16.3%** of users leave the service.
> * The vast majority of continuous features (`Tenure`, `WarehouseToHome`, `NumberOfAddress`, `DaySinceLastOrder`, `CashbackAmount`) show positive skewness (0.69-0.95). Mean values are inflated by long right-hand tails. A lack of normality of the distributions is suspected.
> * The high standard deviation indicates a divergence in customers' experiences, which is a key starting point for studying the relationships between features and churn.

> * By far the largest group are customers with a tenure of 0-1 months. **Over 500 users are brand-new customers.**
> * Most customers have **3 or 4 devices** registered.
> * Sales are strongly dominated by two categories: **laptops and accessories (37.0%, 1,004)** and **mobile phones (36.1%, 980)**, which together account for **73.1%** of all orders.
> * The largest group are customers giving a **neutral rating of 3 (28.9%, 783).** The lowest rating of 1 was given by 21.6% (586), while the highest ratings (4 and 5) together account for 37.7% (1,022).
> * More than half of the customers are **married (51.7%, 1,401)**. Single people make up 31.7% (859) and divorced people 16.7% (452).
> * Most users have **2 to 3 saved delivery addresses** (over 1,200 customers).
> * Most customers (71.7%, 1,945) did not submit any complaint. On the other hand, **28.3% (767) of customers submitted at least one complaint**. A complaint rate above 28% is relatively high and points to potential problems with customer service, delivery or product quality.
> * For **cashback**, the highest concentration of customers is around a **value of 150 (140-160)**.

---

## Step 4: Normality testing

Normality of the distributions was checked in order to select suitable tests for further data analysis in the following steps. Given the number of observations, the D'Agostino-Pearson test, based on skewness and kurtosis, was used. The results were supported by a quantile-quantile plot. For every variable, **the hypothesis of a normal distribution was rejected.**

---

## Step 5: Correlations

Spearman's method was used to study correlations, to work around the lack of normality of the distributions. 

>* The strongest relationship with churn is shown by **customer tenure** (*r* = -0.39), which has a negative correlation. Customers with long tenure leave much less often, which suggests that **the first months after registration may be the critical period.**
>* **Complaints** (*r* = 0.25) may be the next main risk factor. Problems with service or products are associated with the decision to leave.
>* **A higher cashback amount is associated with a lower tendency to leave** (*r* = -0.17) - loyalty programmes presumably retain customers effectively.
>* Interestingly, **a longer time since the last order is associated with customers staying in the service (*r* = -0.16).**
>* The number of registered devices (*r* = 0.11) and the satisfaction rating (*r* = 0.10) show a weak positive correlation with leaving the service.
>* Distance to the warehouse (*r* = 0.06) and the number of saved addresses (*r* = 0.03) show a negligible relationship with churn.

>* No pair of independent variables exceeds the correlation threshold of |*r*| > 0.40. This means **there is no redundancy problem and no need to remove features because of multicollinearity**. This is additionally confirmed by the VIF analysis.

---

## Step 6: Hypothesis testing

For continuous and ordinal variables, the Mann-Whitney U test was used, supported by an effect size. For categorical variables, the chi-square test was used. The analysis was supported by plots.

| Hypothesis | Variable | Test | Statistic | p-value | Effect size | Effect interpretation | Conclusion |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| H1 | `Tenure` | Mann-Whitney U (one-sided) | U = 803,668 | < 0.001 | r_rb = -0.60 | large | Customers who leave have a significantly shorter tenure |
| H2 | `WarehouseToHome` | Mann-Whitney U (one-sided) | U = 451,246 | < 0.001 | r_rb = 0.10 | small | Customers who leave live farther from the warehouse, but the effect is weak |
| H3 | `NumberOfDeviceRegistered` | Mann-Whitney U (one-sided) | U = 419,429 | ≈ 1.000* | r_rb = 0.16 | small | Hypothesis rejected: customers who leave have more devices |
| H3 (additional) | `NumberOfDeviceRegistered` | Chi² | χ² = 44.84 | < 0.001 | Cramér's V = 0.13 | small | Significant relationship, in the opposite direction to the assumed one |
| H4 | `PreferedOrderCat` | Chi² | χ² = 149.25 | < 0.001 | Cramér's V = 0.23 | small | Category is significantly related to churn |
| H5 | `SatisfactionScore` | Mann-Whitney U (one-sided) | U = 421,993 | ≈ 1.000* | r_rb = 0.16 | small | Hypothesis rejected: higher satisfaction does not protect against leaving |
| H5 (additional) | `SatisfactionScore` | Chi² | χ² = 35.61 | < 0.001 | Cramér's V = 0.11 | small | Significant relationship, counter-intuitive direction |
| H6 | `MaritalStatus` | Chi² | χ² = 99.73 | < 0.001 | Cramér's V = 0.19 | small | Significant relationship; married customers leave less often |
| H7 | `NumberOfAddress` | Mann-Whitney U (one-sided) | U = 482,341 | 0.904 | r_rb = 0.04 | negligible | No significant difference; the number of addresses is not related to loyalty |
| H8 | `Complain` | Chi² | χ² = 168.64 | < 0.001 | Cramér's V (= φ) = 0.25 | small | Customers with a complaint leave significantly more often |
| H9 | `DaySinceLastOrder` | Mann-Whitney U (one-sided) | U = 628,801.5 | ≈ 1.000* | r_rb = -0.25 | small | Hypothesis rejected: customers who leave have *fewer* days since their last order |
| H10 | `CashbackAmount` | Mann-Whitney U (one-sided) | U = 633,353.5 | < 0.001 | r_rb = -0.26 | small | Customers who leave have significantly lower cashback |

* Significance level α = 0.05.
* \* In one-sided tests, p ≈ 1.000 means that the difference is in the direction opposite to the alternative hypothesis. This is not an absence of a relationship: in a two-sided test the result would be significant.

* H1: Customers with a shorter tenure (`Tenure`) leave more often than long-term customers.
> * New users have a significantly higher risk of leaving than customers with a longer tenure (large effect). It is suggested to strengthen onboarding and retention activities in the early phase of the customer lifecycle.
  
* H2: A greater distance from the warehouse to the customer's delivery address (`WarehouseToHome`) is associated with a higher churn rate.
> * The distance between the warehouse and home is significantly higher for customers who leave than for those who stay in the service. However, the effect is weak, so the practical importance of this feature is limited.
 
* H3: A larger number of registered devices (`NumberOfDeviceRegistered`) indicates the user's attachment to the service, which reduces the risk of churn in this group.
> * The hypothesis was not confirmed. The number of registered devices is not associated with greater loyalty; on the contrary, the highest share of customers who leave is among customers with the largest number of connected devices.

* H4: Customers whose preferred category (`PreferedOrderCat`) is electronics show different churn behaviour than customers choosing more everyday categories (food, fashion).
> * The preferred category is significantly related to the decision to leave. People buying mainly food almost never leave the service, which cannot be said of users choosing mobile phones.

* H5: A lower customer satisfaction rating (`SatisfactionScore`) is associated with a higher likelihood of leaving.
> * The hypothesis was not confirmed: a high satisfaction rating is not associated with lower churn, and the direction of the relationship is opposite to the expected one. A possible explanation is the co-occurrence of high ratings and complaints.

* H6: Married customers (`MaritalStatus`) leave less often than customers with another marital status.
> * Marital status is significantly related to churn. Married users show greater stability and loyalty to the service, while single users require dedicated retention strategies.

* H7: A larger number of saved addresses (`NumberOfAddress`) indicates the user's attachment to the service, which reduces the risk of churn in this group.
> * The number of addresses saved in the service is not higher for customers who do not decide to leave. It is therefore not related to the stability and loyalty of users and may simply result from convenience.

* H8: Customers submitting complaints (`Complain`) have a significantly higher churn rate than customers who do not submit them.
> * The churn rate differs significantly between customers who submit complaints and customers who have never done so. The strength of this relationship is small, but significant. Submitting a complaint is a strong warning sign, so it is worth introducing measures dedicated to handling complaints efficiently.

* H9: A longer time since the last order (`DaySinceLastOrder`) increases the likelihood that the customer will leave the service.
> * Contrary to expectations, the number of days since the last order is lower in the group of customers who left the service than in the group that stayed (small effect). The result is counter-intuitive and probably reflects characteristics of the dataset rather than real customer behaviour.

* H10: A higher cashback (`CashbackAmount`) is associated with a lower likelihood of churn.
> * Customers who leave have a significantly lower cashback amount than users who stay in the service. This suggests that cashback and purchasing activity may be associated with customer loyalty to the platform, so it is worth considering additional measures to strengthen users' attachment.

---

## Step 7: Analytical dashboard in Power BI

<img width="1302" height="707" alt="Power BI analytical dashboard: KPIs, filters and churn charts by complaints, tenure, category and number of devices" src="https://github.com/user-attachments/assets/6a765bd0-49b4-40fc-bc91-51fdf5272784" />

An operational and analytical report was designed in Power BI, consisting of three main sections:

1. **KPIs**: Number of Customers (2,712), Churn Rate (16.3%), Complaint Rate (28.3%), Average Tenure (10 months), Satisfaction Score (3.0).
2. **Filter panel**: Option to segment by marital status, tenure groups and preferred order category.
3. **Detailed analysis**:
   - Impact of complaints: the Churn Rate is 31.0% for customers who submitted a complaint vs 10.5% for customers without complaints.
   - Impact of tenure: 31.5% of churn occurs among customers with a tenure of 0-6 months. For customers with a tenure above 2 years, the rate drops to 0%.
   - Number of devices: the Churn Rate rises from 6.4% with 2 registered devices to 35.2% with 6 devices.

---

## Step 8: Machine learning model selection

Before modelling, categorical variables were encoded using One-Hot Encoding.

Through stratified 5-fold cross-validation, the training set was divided into 5 equal parts while preserving the proportion of the minority class (customers who leave). This prevents instability of metric estimates in individual folds. To limit the risk of data leakage, all transformation steps, including standardisation of numerical features and generation of synthetic SMOTE samples, are performed exclusively inside the cross-validation loop (*pipeline*), on the training data of a given fold. The exceptions are capping and imputation, which were performed earlier on the whole training set.

The results refer to cross-validation on the training set at a threshold of 0.5.

| Metric | Logistic regression | Random Forest | XGBoost |
| :--- | :---: | :---: | :---: |
| ROC-AUC Score | 0.8819 | 0.9282 | 0.9240 |
| Recall (Churn = 1) | 0.8053 | 0.7194 | 0.6879 |
| Precision (Churn = 1) | 0.4350 | 0.6750 | 0.7009 |
| F1-Score (Churn = 1) | 0.56 | 0.70 | 0.69 |

Three models were compared:
* **Logistic regression**, as the simplest model, serves as a baseline for the tree-based algorithms. It achieves the highest *Recall* (80.53%), which means that it catches the most customers who leave. However, it shows very low *Precision* (43.50%) and the lowest *F1-Score* (0.56). In practice it generates a lot of false alarms, classifying loyal customers as at risk of leaving, which increases the cost of retention campaigns.
* **XGBoost** achieves the highest precision (70.09%), producing the most accurate hits, but at the cost of lower *Recall* (68.79%), i.e. missing a larger number of actual departures. It also has a very high *ROC-AUC* (0.92) and *F1-Score* (0.69).
* **Random Forest** dominates in the main metrics of overall quality: *ROC-AUC* (0.93) and *F1-Score* (0.70). It maintains the most optimal balance between detecting departures (*Recall* = 71.94%) and avoiding false alarms (*Precision* = 67.50%).

***Random Forest*** was selected for the hyperparameter tuning stage. In the context of preventing customer churn, we wanted to minimise both the cost of not taking action towards a customer who leaves (shortfalls in *Recall*) and the cost of unnecessary discounts sent to loyal customers (shortfalls in *Precision*). *Random Forest* provides an **optimal compromise (highest *F1-Score*)**. The highest *ROC-AUC* result indicates a good ability of the model to separate the classes regardless of the chosen cut-off threshold, which is a foundation for the later tuning and adjustment of the decision threshold.

Confusion matrix for the *Random Forest* model:
- True Negative (correctly classified customers who stay): 2116
- True Positive (correctly detected churn): 318
- False Positive (false alarm): 154
- False Negative (missed churn): 124

---

## Step 9: Searching for the best parameters

The best parameters for *Random Forest* were identified:

Split criterion: entropy turned out to be slightly better than the standard Gini in building purer splits in the trees.
Maximum tree depth: the trees can grow without an artificial depth limit, adapting to complex relationships in the data.
Minimum number of samples in a leaf: 1 and in a node: 2.
Number of trees in the forest: the forest consists of 200 decision trees, which ensures stability and high predictive power.

**Assessment of tuning effectiveness**

The best *F1-Score* obtained is 0.71, compared to the initial baseline model, which had 0.70. *Precision* increased from 0.67 to 0.68 and *Recall* from 0.72 to 0.73. Hyperparameter tuning brought a slight improvement in the quality of churn detection.

Confusion matrix for the *Random Forest* model with improved parameters:
- True Negative (correctly classified customers who stay): 2120
- True Positive (correctly detected churn): 323
- False Positive (false alarm): 150
- False Negative (missed churn): 119

## Step 10: Interpretation
* **Feature importance:** Tenure is by far the most important feature (weight above 0.25). This means that the customer's tenure with the company is of fundamental importance for predicting the risk of leaving.
The cashback amount ranks second, which indicates that financial benefits are strongly associated with loyalty. The distance from the warehouse to home and marital status affect the logistics and demographic profile of the customer.
* **Interpretation of logistic regression:** Submitting a complaint increases the customer's chance of leaving by as much as 109.2%. A specific product category (Others) raises the risk of leaving by about 79.5%. A larger number of addresses linked to the account increases the risk of deciding to leave by 74.7% (the result contrasts with the lack of a relationship in the EDA; it may be a conditional effect after accounting for the other features), and a larger number of registered devices causes an increase of about 59.9%. Being single drives the probability of leaving by 48.7%. Days since the last order show a negative effect of -23.8%, which is counter-intuitive and consistent with the earlier analysis.
* **SHAP:** High tenure values cluster on the left side (negative SHAP values), which means that long tenure lowers the risk score. Low tenure values (blue) are shifted to the right, which confirms that new customers are the most exposed to churn. High values (red points) for being single and for preferring phone orders generate positive SHAP values, which raises the risk score. The varied distribution of points for the cashback amount shows that moderate and higher cashback amounts shift the score to the left (lower the risk). In line with the earlier models, the presence of complaints strongly shifts values towards the positive side (right side of the plot) and raises the risk score.

**SHAP and regression coefficients describe relationships in the data, not the causes of churn.**

---

## Step 11: Risk forecast for the new customer base and Power BI dashboard

The selected model was applied to the test data in order to calculate churn risk and indicate at-risk customers. The results were not calibrated, so they may not reflect real probabilities (they may be overestimated). For the purposes of the project, however, they were treated as the probability of churn.

A threshold of 0.35 was adopted, in order to capture as many users as possible who may be inclined to leave the service.

<img width="1322" height="738" alt="Power BI prediction dashboard: customer list by risk score, customer card, risk distribution and SHAP chart" src="https://github.com/user-attachments/assets/3beb2245-c6f7-404b-a6ac-6599077c2deb" />

1. **Indicators:** Among 768 customers, **165 exceed the risk threshold**, which amounts to 21.5% of the whole base. The average risk score among these customers is **71.6%** (uncalibrated).
2. **Risk distribution:** Of the 165 at-risk users, as many as 75 are in the critical risk group (75%-100%).
3. **Customer list and customer card:** customers sorted in descending order by risk score, together with the details of the selected customer.
4. **Filter panel**: Option to segment by marital status, tenure groups, preferred order category, whether a complaint was submitted and satisfaction rating.
5. **SHAP chart:** average impact of features on the model score.

---

## Key findings 

1. **Risk of early churn:**
   - As many as **31.5%** of customers leave within the first **0-6 months** of using the platform.
   - After the 2-year (24-month) mark, the churn rate drops to almost **0%**. Customer loyalty rises dramatically over time.

2. **Complaint as a strong warning sign of leaving:**
   - Customers who submitted a complaint leave more than **3 times as often** (31.0%) compared to people without complaints (10.5%).
   - The complaint shows the strongest positive correlation with the `Churn` variable (+0.25). This is a statistical relationship, not proof that the complaint is the cause of leaving.

3. **Satisfaction paradox:**
   - A high level of declared satisfaction (rating 5/5) **does not protect the customer from leaving**. Among customers with a complaint, churn rises from 19.5% (rating 1) to 40.0% (rating 4) and 35.8% (rating 5), and SHAP assigns a low weight to satisfaction itself.

4. **Number of devices:**
   - As the number of registered devices on the account increases, the *churn* rate rises from **6.4%** (for 2 devices) to as much as **35.2%** (for 6 devices). This may indicate account sharing with other users or problems with the consistency of the user experience (UX) across multiple devices.

5. **Cashback and purchasing activity:**
Higher cashback amounts are associated with lower *churn*. The number of days since the last order is associated with churn in a counter-intuitive direction: the more days since the last order, the lower the *churn*.

---

## Business recommendations

Based on the exploratory analysis and the results of the Random Forest predictive model, 3 main pillars of retention activities were defined:

#### 1. New customer protection programme (protection during the first 0-6 months)
* **Problem:** The churn rate among customers with a tenure of up to 6 months is as high as **31.5%**, and then drops sharply in the following months and years.
* **Action:** 
  * Introduce a dedicated onboarding cycle (e.g. a series of e-mail/push messages with a guide to the benefits of the platform).
  * Grant personalised incentives and discounts (**cashback**) in the 2nd and 4th month after registration, to build a habit of regular purchases.

#### 2. Prioritising complaint handling (SLA & Recovery Program)
* **Problem:** The complaint is one of the strongest signals of leaving - the *churn* rate among people submitting a complaint is **31.0%** (vs 10.5% among the others). In addition, a "satisfaction paradox" was identified, where high ratings (5/5) do not protect against leaving if they are accompanied by a complaint (churn then reaches **35.8%**).
* **Action:**
  * Introduce automatic flagging in the ticketing system (Helpdesk/CRM) for complaints coming from new customers (tenure < 6 months).
  * Shorten the response time (SLA) for these tickets to a maximum of 12-24 hours, and automatically grant compensation (e.g. a discount code or additional cashback) once the ticket is closed.

#### 3. Deploying the Random Forest model in the CRM system (automatic Churn Risk alerts)
* **Problem:** Reacting only when a customer stops buying is usually too late.
* **Action:**
  * Integrate the trained Random Forest model with the customer database to generate a weekly churn probability score (*Churn Probability*).
  * **Automation of retention actions:**
    * **High risk (Probability > 75%):** Automatically pass the record to the marketing/retention team for direct contact with an individual offer.
    * **Medium risk (Probability 35%-75%):** An automatic trigger in the system sending a personalised discount coupon or an offer of higher cashback on the preferred product category.
   
---

## Limitations

1. **Capping and imputation before cross-validation:** The capping bounds (IQR) and the medians used to fill missing values were determined on the whole training set before cross-validation, rather than inside the pipeline. This causes a minor information leak between folds. The test set is not affected, because it uses values computed exclusively on the training set.

2. **No probability calibration:** Model outputs are treated in the project as churn probabilities, but they were not calibrated and the model was trained with SMOTE. The score distribution is strongly polarised. They should therefore be interpreted primarily as a risk ranking of customers, not as real chances of leaving.

3. **Decision threshold:** The adopted threshold of 0.35 was selected on out-of-fold predictions from cross-validation and justified by the cost of retention actions. However, no detailed cost analysis (cost of a retention action vs value of a lost customer) was carried out, due to a lack of such data.

4. **No evaluation of the model on the new customer base:** The test set does not contain information about customers' actual decisions and serves as a simulation of a new base. The prediction dashboard shows predictions, not measured performance. Model quality was estimated solely on the basis of cross-validation on the training set. In a real deployment, monitoring would be necessary (comparing predictions with actual churn).

5. **Associations, not causes:** The analysis of correlations, hypothesis tests, regression coefficients and SHAP values shows relationships between features and churn, not causal relationships. Some results are counter-intuitive (e.g. higher satisfaction and fewer days since the last order are associated with more frequent churn) and probably reflect characteristics of the dataset rather than real customer behaviour.

6. **Dataset:** The project is based on a single public dataset (Kaggle) of limited size (3,941 observations before cleaning). The dataset contains no customer identifier, so duplicate removal relies on identical values across all features. Customer names on the prediction dashboard are synthetic.

7. **Hyperparameter tuning:** The improvement after tuning Random Forest was small (F1-Score improved from 0.70 to 0.71) and probably falls within the fold-to-fold variability, so it should not be interpreted as a significant difference.

8. **Multiple hypothesis testing:** With 10 tests, no correction for multiple testing was applied.

---

## Next steps

- Evaluate the calibration curve on out-of-fold predictions and apply calibration, so that the outputs can be interpreted as probabilities.
- Move capping and imputation into the cross-validation pipeline.
- Choose the threshold based on a cost analysis and a recall/precision table for different thresholds.
- Add the main risk drivers for each customer to the dashboard.
- Monitor the model after deployment: compare predictions with actual churn, track data drift, retrain periodically.
- Verify the findings on another dataset or on real-world data.
