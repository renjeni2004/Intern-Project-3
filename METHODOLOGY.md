# Methodology

## Problem framing
Customer churn is a binary classification problem: estimate whether a customer is likely to discontinue a service. The output supports prioritization for retention review; it does not establish why a customer leaves.

## Workflow
1. Load the customer-level CSV and inspect data types and missingness.
2. Remove `customerID`, because it is an identifier rather than a customer behavior feature.
3. Map `Churn` to a binary target.
4. Engineer `AvgSpendPerTenureMonth = TotalCharges / tenure`; zero-tenure values are treated as missing.
5. Split into stratified training and test partitions (80/20).
6. Fit imputation, scaling and one-hot encoding within a pipeline.
7. Compare Logistic Regression and Random Forest.
8. Run stratified 5-fold cross-validation on the training partition.
9. Tune a Random Forest with RandomizedSearchCV.
10. Select the final pipeline using held-out ROC-AUC comparison and persist it with Joblib.
11. Evaluate with multiple metrics and diagnostic plots.
12. Run K-Means (k=3) on standardized tenure, monthly charges and total charges.

## Leakage controls
Preprocessing is part of the estimator pipeline, so imputation statistics, scaling parameters and category vocabularies are learned from each training fold rather than the full dataset. The test partition is held out from cross-validation and model fitting. The test partition is used for final model selection in this educational implementation; for a stricter production-grade workflow, use a separate validation set for model/threshold selection and reserve the test set for one final evaluation.

## Feature selection
Permutation importance is calculated on the held-out partition using ROC-AUC decrease. It is a model-dependent diagnostic, not proof of causality. Correlated features can share or dilute importance.

## Clustering
K-Means uses three numeric measures after standardization. Silhouette score is saved as a descriptive internal cluster metric. Cluster IDs are arbitrary and should be profiled before assigning business labels.
