
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT/"models/churn_pipeline.joblib"
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")
st.title("Customer Churn Prediction")
st.caption("Machine Learning | Customer Retention Analytics")
if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python src/train.py` from the project root first.")
    st.stop()
model = joblib.load(MODEL_PATH)

with st.sidebar:
    st.header("Customer profile")
    tenure = st.slider("Tenure (months)",0,72,12)
    monthly = st.number_input("Monthly charges", min_value=0.0, max_value=200.0, value=75.0, step=1.0)
    total = st.number_input("Total charges", min_value=0.0, max_value=20000.0, value=float(tenure*monthly), step=10.0)
    contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])
    internet = st.selectbox("Internet service",["DSL","Fiber optic","No"])
    support = st.selectbox("Tech support",["Yes","No","No internet service"] if internet=="No" else ["Yes","No"])
    security = st.selectbox("Online security",["Yes","No","No internet service"] if internet=="No" else ["Yes","No"])
    payment = st.selectbox("Payment method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    gender = st.selectbox("Gender",["Female","Male"])
    senior = st.selectbox("Senior citizen",[0,1])
    partner = st.selectbox("Partner",["Yes","No"])
    dependents = st.selectbox("Dependents",["Yes","No"])
    phone = st.selectbox("Phone service",["Yes","No"])
    backup = st.selectbox("Online backup",["Yes","No","No internet service"] if internet=="No" else ["Yes","No"])
    protection = st.selectbox("Device protection",["Yes","No","No internet service"] if internet=="No" else ["Yes","No"])
    tv = st.selectbox("Streaming TV",["Yes","No","No internet service"] if internet=="No" else ["Yes","No"])
    movies = st.selectbox("Streaming movies",["Yes","No","No internet service"] if internet=="No" else ["Yes","No"])
    paperless = st.selectbox("Paperless billing",["Yes","No"])

input_row = pd.DataFrame([{
    "SeniorCitizen":senior,"Partner":partner,"Dependents":dependents,"tenure":tenure,
    "PhoneService":phone,"InternetService":internet,"OnlineSecurity":security,
    "OnlineBackup":backup,"DeviceProtection":protection,"TechSupport":support,
    "StreamingTV":tv,"StreamingMovies":movies,"Contract":contract,
    "PaperlessBilling":paperless,"PaymentMethod":payment,"MonthlyCharges":monthly,
    "TotalCharges":total,"gender":gender,
    "AvgSpendPerTenureMonth": total/tenure if tenure>0 else None
}])
st.subheader("Prediction")
if st.button("Predict churn risk", type="primary"):
    probability = float(model.predict_proba(input_row)[0,1])
    prediction = int(probability >= 0.5)
    c1,c2 = st.columns(2)
    c1.metric("Estimated churn probability",f"{probability:.1%}")
    c2.metric("Prediction","Likely to churn" if prediction else "Likely to stay")
    st.progress(probability)
    if probability >= 0.5:
        st.warning("Higher-risk customer. Review contract, service issues, support experience and pricing.")
    else:
        st.success("Lower estimated churn risk. Continue monitoring customer experience.")
    st.caption("This is a model estimate, not a guarantee or a substitute for business judgment.")
st.divider()
st.caption("Educational project. The included training data is synthetic and does not represent actual customers.")
