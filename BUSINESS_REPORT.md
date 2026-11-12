# Business & Model Report

## Executive summary
This project demonstrates a customer churn classification workflow using a reproducible synthetic dataset of **5,000 customer records**. The final saved pipeline is **Logistic Regression**. The held-out evaluation results are listed below.

> **Critical caveat:** These data are synthetic. The model metrics are software/workflow validation results, not evidence of real-world telecom churn performance. No actual customer or commercial dataset was supplied.

## Dataset profile
- Records: 5,000
- Input columns before feature engineering: 19
- Target: Churn
- Positive class rate in the generated data: 22.5%
- Missingness was intentionally introduced in numeric charge columns to demonstrate imputation.

## Model comparison
| model               |   cv_f1_mean |   cv_roc_auc_mean |   accuracy |   precision |   recall |    f1 |   roc_auc |
|:--------------------|-------------:|------------------:|-----------:|------------:|---------:|------:|----------:|
| Logistic Regression |        0.542 |             0.794 |      0.704 |       0.413 |    0.751 | 0.533 |     0.797 |
| Random Forest       |        0.451 |             0.776 |      0.799 |       0.567 |    0.453 | 0.504 |     0.791 |

## Final model metrics
| Metric | Held-out test result |
|---|---:|
| Accuracy | 0.704 |
| Precision | 0.413 |
| Recall | 0.751 |
| F1-score | 0.533 |
| ROC-AUC | 0.797 |

Interpretation:
- Recall estimates the proportion of actual churners identified by the classifier.
- Precision estimates the proportion of predicted churners who are actual churners.
- F1-score balances precision and recall.
- ROC-AUC measures ranking discrimination across thresholds.
- Accuracy alone can conceal poor minority-class performance.

## Feature importance
Permutation importance is provided in `permutation_feature_importance.csv` and plotted in `figures/feature_importance.png`. Interpret it as model-specific predictive contribution, not causal impact.

## Customer segmentation
K-Means was run with k=3 on standardized tenure, monthly charges and total charges. Segment profiles are in `customer_segments.csv`; the silhouette score is in `clustering_summary.json`. Segment numbers are arbitrary identifiers.

## Business recommendations (hypotheses to validate)
1. Review short-tenure customers and month-to-month contracts as potential retention cohorts.
2. Examine support and security-service experience where customer feedback or operational data confirms service friction.
3. Test retention offers with controlled experiments; do not assume a discount will reduce churn.
4. Track precision, recall, intervention cost, retained customer value and campaign uplift together.

## Limitations and next steps
- Replace synthetic data with a licensed/authorized real dataset.
- Use a temporal validation strategy and a separate untouched test set.
- Calibrate probabilities and optimize the decision threshold against intervention economics.
- Monitor drift, fairness, privacy, and model performance after deployment.
