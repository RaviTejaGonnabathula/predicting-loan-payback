# **Predictive Default Modeling: A Soft-Voting GBM Ensemble for Financial Risk**

## **Project Objective**

This repository presents a high-performance, end-to-end Machine Learning solution for *binary classification* in the *credit risk* domain. The core objective is to predict the likelihood of a loan being *paid back* versus *defaulting*. This modeling framework is essential for optimizing *lending strategies*, setting *capital requirements*, and assessing *portfolio health*.

The project leverages an ensemble of state-of-the-art *Gradient Boosting Machines (GBMs)*, fine-tuned through automated *hyperparameter optimization*, and supported by advanced *feature engineering* grounded in financial domain logic.

**Target Variable:** *loan_paid_back*  
*(Binary: 0 = Default / Non-Payback, 1 = Paid Back)*

---

## **Libraries and Their Purpose**

| **Category**      | **Library / Module**                   | **Purpose** |
|-------------------|-----------------------------------------|-------------|
| **Core**          | `numpy`, `pandas`                       | High-efficiency data manipulation and *numerical operations*. |
| **Modeling**      | `lightgbm`, `xgboost`, `catboost`       | Primary ensemble classifiers (*Gradient Boosting Machines*). |
| **Optimization**  | `optuna`                                | *Bayesian Hyperparameter Optimization (HPT)* framework. |
| **Explainability**| `SHAP` *(SHapley Additive exPlanations)*| Model-agnostic interpretation of *feature importance* and *local predictions*. |
| **Metrics**       | `sklearn.metrics`                       | Comprehensive evaluation including *AUC*, *Gini*, and *F1-score*. |
| **Diagnostics**   | `statsmodels`                           | Statistical tools for advanced checks (e.g., *Variance Inflation Factor*). |
| **Visualization** | `seaborn`, `matplotlib`                | Data profiling, *correlation matrices*, and model diagnostic plots. |

---

## **Methodology & Computational Pipeline**

The pipeline consists of four rigorous, sequential phases ensuring statistical reliability and maximum predictive power.

---

## **Phase I: Data Quality & Exploratory Data Analysis (EDA)**

This phase validates data integrity and identifies inherent relationships within the dataset.

- **Initial Audit:** Conducted thorough checks for *missing values*, *duplicates*, and metadata profiling using `.info()` and `.describe()`.
- **Target Imbalance Check:** Evaluated the distribution of *loan_paid_back* to confirm class imbalance, guiding the use of *AUC* over simple *Accuracy*.
- **Outlier & Skewness Profiling:** Assessed *skewness*, *zero-inflation*, and *IQR-based outliers* across all numerical features.
- **Multicollinearity Assessment:** Computed *Pearson* and *Spearman* correlation matrices. Additionally, *Variance Inflation Factor (VIF)* was calculated to detect multicollinearity (features with *VIF > 10* were flagged).  
 

---

## **Phase II: Domain-Driven Feature Engineering**

Raw features were enhanced using principles from financial risk analytics to maximize class separation.

### **Robust Outlier Mitigation**
- **Clipping & Flagging:** Critical variables were clipped to plausible bounds.  
  Example: *credit_score* was bounded to **[300, 900]**, and a *flag* feature captured out-of-range values.
- **DTI Sanitization:** The *debt_to_income_ratio* was clipped to a realistic domain (e.g., **[0, 2]**) to reduce distortion from extreme values.  
 

### **Feature Synthesis**
New domain-informed features were engineered, including:

- *income_loan_ratio*  
- *affordability_index*  
- *risk_margin* *(derived from income, tenure, rate)*  
  _[cite: uploaded:predicting-loan-payback]_  
- *Log transformations* applied to highly skewed variables such as *interest_rate* and *annual_income*.

### **Encoding**
Categorical features were encoded using *LabelEncoder*, which is optimal for *tree-based* models.

---

## **Phase III: Model Training and Optimization**

The ensemble architecture reduces bias and variance, ensuring resilient generalization.

- **Baseline Modeling:** Default-parameter models (LightGBM, XGBoost, CatBoost) were trained to establish initial performance benchmarks.
- **Automated Hyperparameter Tuning:**  
  *Optuna* performed *Bayesian/TPE* optimization across all GBMs with the objective of maximizing *AUC* on validation folds.  


- **Soft-Voting Ensemble:**  
  Final predictions are the *mean* of probabilities from the optimized GBMs. This *soft-voting* strategy yields superior stability and robustness.  


---

## **Phase IV: Evaluation and Interpretability**

Model performance is measured using financial-grade metrics and supplemented with transparent explainability.

### **Key Performance Indicators**
- **Primary Metric:** *AUC (Area Under ROC Curve)*  
- **Financial Metrics:**  
  - *Gini Coefficient* → \(2 \times AUC - 1\)  
  - *Kolmogorov-Smirnov (KS) Statistic*

### **Diagnostic Visualization**
Generated:
- *ROC Curves*
- *Precision-Recall Curves*
- Threshold-based performance profiles

### **SHAP Interpretation**
*SHAP values* were used to decompose individual predictions, revealing the *magnitude* and *direction* of each feature's contribution. This is essential for *regulatory explainability*, *risk review boards*, and *model governance*.  


---

