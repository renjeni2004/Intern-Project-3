
"""Train and evaluate the Customer Churn Prediction models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/customer_churn_synthetic.csv"
MODEL_DIR = ROOT / "models"; MODEL_DIR.mkdir(exist_ok=True)
FIG_DIR = ROOT / "reports/figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = ROOT / "reports"; REPORT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

def main():
    df = pd.read_csv(DATA)
    # Remove identifier: it has no predictive meaning.
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    # Feature engineering: average monthly spend per active tenure month.
    df["AvgSpendPerTenureMonth"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
    df["AvgSpendPerTenureMonth"] = df["AvgSpendPerTenureMonth"].replace([np.inf, -np.inf], np.nan)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=350, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv,
            scoring={"f1":"f1","roc_auc":"roc_auc","accuracy":"accuracy"}, n_jobs=-1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:,1]
        rows.append({"model":name, "cv_f1_mean":scores["test_f1"].mean(),
            "cv_roc_auc_mean":scores["test_roc_auc"].mean(),
            "accuracy":accuracy_score(y_test,pred),
            "precision":precision_score(y_test,pred,zero_division=0),
            "recall":recall_score(y_test,pred,zero_division=0),
            "f1":f1_score(y_test,pred,zero_division=0),
            "roc_auc":roc_auc_score(y_test,proba)})
        fitted[name] = pipe

    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    comparison.to_csv(REPORT_DIR/"model_comparison.csv", index=False)
    best_name = comparison.iloc[0]["model"]
    best_model = fitted[best_name]

    # Hyperparameter tuning for the selected model family (Random Forest).
    rf_pipe = Pipeline([("preprocessor", preprocessor),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))])
    param_dist = {"model__n_estimators":[200,300,400],
                  "model__max_depth":[None,8,14,20],
                  "model__min_samples_split":[2,5,10],
                  "model__min_samples_leaf":[1,2,4]}
    search = RandomizedSearchCV(rf_pipe, param_distributions=param_dist, n_iter=8,
        scoring="roc_auc", cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    tuned = search.best_estimator_
    tuned_pred = tuned.predict(X_test)
    tuned_prob = tuned.predict_proba(X_test)[:,1]
    tuned_metrics = {
        "selected_model_before_tuning": best_name,
        "tuned_model": "Random Forest",
        "best_params": search.best_params_,
        "test_accuracy": accuracy_score(y_test,tuned_pred),
        "test_precision": precision_score(y_test,tuned_pred,zero_division=0),
        "test_recall": recall_score(y_test,tuned_pred,zero_division=0),
        "test_f1": f1_score(y_test,tuned_pred,zero_division=0),
        "test_roc_auc": roc_auc_score(y_test,tuned_prob),
        "classification_report": classification_report(y_test,tuned_pred,output_dict=True,zero_division=0)
    }
    # Keep tuned RF only if its held-out ROC-AUC is at least as good as the baseline best.
    base_best_auc = comparison.iloc[0]["roc_auc"]
    if tuned_metrics["test_roc_auc"] >= base_best_auc:
        final_model = tuned
        final_pred, final_prob = tuned_pred, tuned_prob
        final_name = "Tuned Random Forest"
    else:
        final_model = best_model
        final_pred = best_model.predict(X_test)
        final_prob = best_model.predict_proba(X_test)[:,1]
        final_name = best_name
    joblib.dump(final_model, MODEL_DIR/"churn_pipeline.joblib")
    (REPORT_DIR/"metrics.json").write_text(json.dumps({
        "final_model": final_name,
        "final_test_accuracy": accuracy_score(y_test,final_pred),
        "final_test_precision": precision_score(y_test,final_pred,zero_division=0),
        "final_test_recall": recall_score(y_test,final_pred,zero_division=0),
        "final_test_f1": f1_score(y_test,final_pred,zero_division=0),
        "final_test_roc_auc": roc_auc_score(y_test,final_prob),
        "tuning": tuned_metrics,
        "data_note": "Synthetic educational dataset; not real customer records."
    }, indent=2))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay.from_predictions(y_test, final_pred, display_labels=["No churn","Churn"], ax=ax, cmap="Blues")
    ax.set_title("Customer Churn — Confusion Matrix")
    fig.tight_layout(); fig.savefig(FIG_DIR/"confusion_matrix.png", dpi=180); plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(6,5))
    RocCurveDisplay.from_predictions(y_test, final_prob, ax=ax)
    ax.set_title("Customer Churn — ROC Curve")
    fig.tight_layout(); fig.savefig(FIG_DIR/"roc_curve.png", dpi=180); plt.close(fig)

    # Model comparison
    fig, ax = plt.subplots(figsize=(7,4))
    comparison.set_index("model")[["f1","roc_auc"]].plot(kind="bar", ax=ax)
    ax.set_ylim(0,1); ax.set_ylabel("Score"); ax.set_title("Baseline Model Comparison")
    ax.legend(["F1 score","ROC-AUC"]); fig.tight_layout()
    fig.savefig(FIG_DIR/"model_comparison.png", dpi=180); plt.close(fig)

    # Permutation importance on held-out data, using the complete pipeline.
    perm = permutation_importance(final_model, X_test, y_test, n_repeats=8,
        random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1)
    imp = pd.DataFrame({"feature":X.columns, "importance_mean":perm.importances_mean,
        "importance_std":perm.importances_std}).sort_values("importance_mean",ascending=False)
    imp.to_csv(REPORT_DIR/"permutation_feature_importance.csv",index=False)
    fig, ax = plt.subplots(figsize=(8,5))
    top = imp.head(10).sort_values("importance_mean")
    ax.barh(top["feature"],top["importance_mean"],xerr=top["importance_std"])
    ax.set_title("Top 10 Permutation Feature Importances"); ax.set_xlabel("Decrease in ROC-AUC")
    fig.tight_layout(); fig.savefig(FIG_DIR/"feature_importance.png",dpi=180); plt.close(fig)

    # K-Means segmentation on selected numeric customer attributes.
    seg = df[["tenure","MonthlyCharges","TotalCharges"]].copy()
    seg = seg.fillna(seg.median())
    seg_scaled = StandardScaler().fit_transform(seg)
    km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(seg_scaled)
    df["CustomerSegment"] = labels
    df.groupby("CustomerSegment").agg(
        customers=("CustomerSegment","size"),
        avg_tenure=("tenure","mean"),
        avg_monthly_charges=("MonthlyCharges","mean"),
        churn_rate=("Churn",lambda s:(s==1).mean())
    ).to_csv(REPORT_DIR/"customer_segments.csv")
    sil = silhouette_score(seg_scaled, labels)
    (REPORT_DIR/"clustering_summary.json").write_text(json.dumps({
        "algorithm":"K-Means","k":3,"silhouette_score":sil,
        "features":["tenure","MonthlyCharges","TotalCharges"],
        "note":"Exploratory segmentation on scaled numeric attributes."
    },indent=2))

    print("Training complete.")
    print(comparison.to_string(index=False))
    print("Final model:", final_name)
    print("Metrics:", json.dumps({k:v for k,v in tuned_metrics.items() if k.startswith("test_")}, indent=2))

if __name__ == "__main__":
    main()
