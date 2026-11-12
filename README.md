 ##Customer Churn Prediction System

**Sumerix Global Internship — Data Science & Data Analytics Using Python | Task 3**

A reproducible machine-learning project that estimates customer churn risk, compares classification models, performs exploratory customer segmentation, and exposes predictions through a Streamlit interface.

> **Important data limitation:** The included `customer_churn_synthetic.csv` is a generated educational dataset, not a real telecom-company dataset and not the original IBM Telco dataset. Results demonstrate an end-to-end workflow only. Do not present the reported metrics as real-world business performance.

## Project objectives
- Clean and inspect customer data; handle missing numeric values.
- Engineer an average-spend feature and preprocess numeric/categorical columns.
- Train and compare Logistic Regression and Random Forest classifiers.
- Evaluate with accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix and ROC curve.
- Apply stratified 5-fold cross-validation and randomized hyperparameter search.
- Save the final preprocessing + estimator pipeline with Joblib.
- Perform K-Means segmentation on selected numeric customer attributes.
- Serve individual predictions with Streamlit.

## Folder structure
```text
customer_churn_prediction_task3/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/customer_churn_synthetic.csv
│   └── processed/
├── docs/
│   ├── BUSINESS_REPORT.md
│   └── METHODOLOGY.md
├── linkedin/
│   ├── LINKEDIN_POST.md
│   ├── linkedin_project_overview.png
│   ├── linkedin_model_results.png
│   └── linkedin_dashboard_preview.png
├── models/churn_pipeline.joblib
├── notebooks/Customer_Churn_Analysis.ipynb
├── reports/
│   ├── figures/
│   ├── clustering_summary.json
│   ├── customer_segments.csv
│   ├── metrics.json
│   ├── model_comparison.csv
│   └── permutation_feature_importance.csv
├── src/
│   └── train.py
├── requirements.txt
└── README.md
```

## 1. Environment setup (VS Code)
Use Python 3.10–3.13. From the project root:

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Train models and generate reports
```bash
python src/train.py
```

This creates/updates:
- `models/churn_pipeline.joblib`
- `reports/model_comparison.csv`
- `reports/metrics.json`
- `reports/customer_segments.csv`
- `reports/permutation_feature_importance.csv`
- diagnostic figures in `reports/figures/`

The script uses a stratified 80/20 train-test split. Imputation, scaling and one-hot encoding are fitted inside a Scikit-learn Pipeline using training data only, reducing preprocessing leakage. Cross-validation is performed on the training split. The held-out test split is used for final evaluation.

## 3. Launch the Streamlit application
```bash
streamlit run app/streamlit_app.py
```
Open the local URL shown in the terminal (normally `http://localhost:8501`). Enter a customer profile and click **Predict churn risk**.

## 4. Open the notebook
```bash
jupyter notebook notebooks/Customer_Churn_Analysis.ipynb
```
The notebook provides an exploratory walkthrough and links to the reproducible training script.

## Dataset and target
- One row represents one synthetic customer.
- Target: `Churn` (`Yes` / `No`).
- Customer identifier is excluded from model training.
- Numeric missing values are imputed with the median.
- Categorical missing values are imputed with the most frequent value.
- Categorical variables are one-hot encoded; numeric variables are standardized.

## Model evaluation
The generated `reports/metrics.json` and `reports/model_comparison.csv` contain the metrics from the actual run on the included synthetic dataset. Do not copy metrics into a public portfolio claim without stating that the dataset is synthetic.

For churn problems, accuracy alone is insufficient. Recall is important when missing a customer who will churn is costly; precision matters when retention interventions are expensive. ROC-AUC measures ranking quality across thresholds. The operational threshold should be selected with business costs and validation data, not assumed to be 0.50 forever.

## Important limitations
1. The dataset is synthetic and encodes deliberately generated relationships. Metrics are not evidence of performance on real customers.
2. The dashboard is a demonstration tool, not a production retention platform.
3. Churn probability is an estimate, not a causal explanation.
4. K-Means clusters are exploratory; cluster labels do not have inherent business meaning.
5. Before deployment, validate on a real, authorized, representative dataset; assess calibration, fairness, drift, privacy, and intervention economics.

## Suggested next steps
- Replace the synthetic CSV with an authorized real churn dataset using the same column schema.
- Establish a temporal holdout to reflect future-customer prediction.
- Tune the decision threshold against retention costs and customer lifetime value.
- Add probability calibration, SHAP explanations, monitoring, and deployment controls.
- Add PostgreSQL ingestion and Power BI/Tableau reporting in the later capstone.

## Deliverables
Python source, dataset, notebook, trained model, Streamlit app, metrics, visualizations, business report, methodology, and LinkedIn-ready assets are included.
