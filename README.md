Predictive Default Modeling: A Soft-Voting GBM Ensemble for Financial Risk

Project Objective

This repository presents a high-performance, end-to-end Machine Learning solution for binary classification in the credit risk domain. The core objective is to predict the likelihood of a loan being paid back versus defaulting. This modeling approach is crucial for optimizing lending strategies, setting capital requirements, and assessing portfolio health.

The project leverages an ensemble of state-of-the-art Gradient Boosting Machines (GBMs), meticulously fine-tuned using automated hyperparameter optimization, and coupled with advanced feature engineering rooted in financial domain knowledge.

Target Variable: loan_paid_back (Binary: 0 = Default/Non-Payback, 1 = Paid Back)

Libraries and Their Purpose

| **Category**      | **Library / Module**                   | **Purpose** |
|-------------------|-----------------------------------------|-------------|
| **Core**          | `numpy`, `pandas`                       | High-efficiency data manipulation and numerical operations. |
| **Modeling**      | `lightgbm`, `xgboost`, `catboost`       | Primary ensemble classifiers (Gradient Boosting Machines). |
| **Optimization**  | `optuna`                                | Bayesian Hyperparameter Optimization (HPT) framework. |
| **Explainability**| `SHAP` (SHapley Additive exPlanations)  | Model-agnostic interpretation of feature importance and local predictions. |
| **Metrics**       | `sklearn.metrics`                       | Comprehensive evaluation including AUC, Gini, and F1-score. |
| **Diagnostics**   | `statsmodels`                           | Statistical tools for advanced checks (e.g., Variance Inflation Factor). |
| **Visualization** | `seaborn`, `matplotlib`                | Data profiling, correlation matrices, and model diagnostic plots. |

Methodology & Computational Pipeline

The pipeline is structured into four sequential, rigorous phases, ensuring both predictive power and statistical soundness.

Phase I: Data Quality & Exploratory Data Analysis (EDA)

This phase ensures the integrity of the data and identifies intrinsic relationships.

Initial Audit: Performed exhaustive checks for missing values (.isnull()), duplicates, and basic metadata profiling (.info(), .describe()).

Target Imbalance Check: Analyzed the class distribution of the target variable (loan_paid_back) to confirm class imbalance and inform the choice of evaluation metrics (prioritizing AUC over Accuracy).

Outlier & Skewness Profiling: Quantified skewness, zero-inflation, and IQR-based outlier counts across all numerical features.

Multicollinearity Assessment: Calculated both Pearson (linear) and Spearman (rank-based) correlation matrices. Critically, the Variance Inflation Factor (VIF) was computed for all feature groups to identify features contributing to high collinearity (VIF > 10) [cite: uploaded:predicting-loan-payback.ipynb].

Phase II: Domain-Driven Feature Engineering

Raw features were transformed and augmented using principles derived from financial risk modeling to maximize predictive separation.

Robust Outlier Mitigation:

Clipping & Flagging: Implemented domain-constrained clipping for critical metrics. For instance, credit_score was capped to its statistical domain [300, 900], with an associated binary flag indicating values outside this range.

DTI Sanitization: The Debt-to-Income Ratio (debt_to_income_ratio) was clipped to a realistic bound (e.g., [0, 2]) to mitigate noise from extreme values [cite: uploaded:predicting-loan-payback (1).ipynb].

Feature Synthesis: A core focus was the creation of novel interaction and ratio features to capture complex financial stress. Key synthesized features include:

income_loan_ratio

affordability_index

risk_margin (Leveraging income, loan tenure, and interest rate) [cite: uploaded:predicting-loan-payback (1).ipynb]

log_transforms applied to highly skewed variables (interest_rate, annual_income).

Encoding: All categorical features were processed using LabelEncoder for optimization with tree-based models.

Phase III: Model Training and Optimization

The ensemble strategy minimizes model bias and variance, yielding highly robust predictions.

Baseline Modeling: Initial training runs were conducted using default parameters for LightGBM, XGBoost, and CatBoost to establish performance baselines.

Automated HPT: The Optuna framework was deployed to conduct efficient hyperparameter optimization (Bayesian/TPE sampling) for all three GBMs. The objective function was strictly defined to maximize the Area Under the ROC Curve (AUC) on the validation fold [cite: uploaded:predicting-loan-payback.ipynb].

Soft-Voting Ensemble: The final probability output is derived from the arithmetic mean of the predicted probabilities from the three best-optimized models. This soft-voting approach enhances generalization and stability [cite: uploaded:predicting-loan-payback (1).ipynb].

Phase IV: Evaluation and Interpretability

Model success is measured against industry-standard financial metrics, and decisions are fully transparent.

Key Performance Indicators (KPIs):

Primary: AUC (Area Under the ROC Curve).

Financial Benchmarks: Gini Coefficient ($2 \times \text{AUC} - 1$) and the Kolmogorov-Smirnov (KS) Statistic.

Diagnostic Visualization: Generated ROC Curves and Precision-Recall Curves to visualize trade-offs between sensitivity and specificity.

SHAP Interpretation: Model decisions are deconstructed using SHAP values to determine the magnitude and direction (positive or negative) of each feature's contribution to a specific loan outcome. This provides the necessary explainability for risk review boards and compliance purposes [cite: uploaded:predicting-loan-payback (1).ipynb].
