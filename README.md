# Credit Risk Modeling Using Machine Learning

## Overview
This project focuses on developing predictive models for credit risk assessment using machine learning techniques. The goal is to predict the credit approval status (Approved_Flag) for prospects based on demographic, financial, and credit-related features.

## Dataset
### Case Study 1
- **File:** `case_study1.xlsx`
- **Description:** Initial dataset containing demographic and financial features of prospects.
- **Preprocessing:** Removed rows where `Age_Oldest_TL` was `-99999`.

### Case Study 2
- **File:** `case_study2.xlsx`
- **Description:** Dataset with additional credit-related features.
- **Preprocessing:** Removed columns with more than 10,000 `-99999` values and rows with `-99999` values.

### Merging Datasets
Merged both datasets on `PROSPECTID` after preprocessing to create a comprehensive dataset for analysis.

## Exploratory Data Analysis (EDA)
- Analyzed distributions and statistics of numerical features (`describe()` function).
- Investigated categorical features (`MARITALSTATUS`, `EDUCATION`, `GENDER`, `last_prod_enq2`, `first_prod_enq2`) using chi-square tests for association with `Approved_Flag`.

## Feature Selection
- Used Variance Inflation Factor (VIF) for numerical features to handle multicollinearity.
- Applied ANOVA test to select categorical features significantly associated with `Approved_Flag`.

## Data Preprocessing
- **Encoding:**
  - Ordinal Encoding for `EDUCATION`.
  - One-Hot Encoding for categorical features (`MARITALSTATUS`, `GENDER`, `last_prod_enq2`, `first_prod_enq2`).
- **Scaling:**
  - Standard Scaling applied to numerical features (`Age_Oldest_TL`, `Age_Newest_TL`, etc.).

## Model Building
### Random Forest Classifier
- Utilized `RandomForestClassifier` with 200 estimators.
- Evaluated model performance metrics: accuracy, precision, recall, and F1-score for each class (P1, P2, P3, P4).

### XGBoost Classifier
- Implemented `XGBClassifier` with hyperparameter tuning (learning_rate, max_depth, n_estimators).
- Assessed model performance metrics: accuracy, precision, recall, and F1-score for each class (P1, P2, P3, P4).

### Decision Tree Classifier
- Deployed `DecisionTreeClassifier` with parameters for max depth and min samples split.
- Reviewed model performance metrics: accuracy, precision, recall, and F1-score for each class (P1, P2, P3, P4).

## Conclusion
- Identified XGBoost classifier as the top-performing model with an accuracy of XX% after hyperparameter tuning.
- Suggested further improvements through feature engineering, ensemble methods, and deployment in a production environment for real-time credit risk assessment.

## Future Work
- Enhance model robustness and accuracy through advanced techniques.
- Explore ensemble methods (e.g., stacking, boosting) for further performance gains.
- Implement the model into a production system for ongoing credit risk monitoring and assessment.

