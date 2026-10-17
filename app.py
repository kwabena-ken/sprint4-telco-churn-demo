import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Telco Churn Demo", page_icon="📡")
st.title("Telco Customer Churn Predictor")
st.write("Enter a few customer details. The model estimates whether they are likely to churn.")

# Load model once
@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")

model = load_model()

# --- user inputs (simplified) ---
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone service", ["Yes", "No"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])

with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly = st.number_input("Monthly charges", min_value=0.0, value=70.0)
    total = st.number_input("Total charges", min_value=0.0, value=800.0)

# Extra service flags
st.subheader("Add-on services")
online_sec = st.selectbox("Online security", ["Yes", "No", "No internet service"])
online_bak = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
device = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
tech = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
stream_mov = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
multi = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])

if st.button("Predict churn"):
    # engineered features (same ideas as Week 3)
    tenure_years = tenure / 12
    long_contract = 1 if contract in ["One year", "Two year"] else 0
    service_yes = sum(
        1 for x in [online_sec, online_bak, device, tech, stream_tv, stream_mov] if x == "Yes"
    )
    avg_charge = total / tenure if tenure > 0 else monthly

    row = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bak,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_mov,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "tenure_years": tenure_years,
        "num_services": service_yes,
        "long_contract": long_contract,
        "avg_charge_per_tenure": avg_charge,
    }])

    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    # assume classes_ order; map safely
    classes = list(model.classes_)
    yes_idx = classes.index("Yes") if "Yes" in classes else 1
    churn_prob = proba[yes_idx]

    if pred == "Yes":
        st.error(f"Prediction: **Likely to churn** (probability ≈ {churn_prob:.0%})")
    else:
        st.success(f"Prediction: **Likely to stay** (churn probability ≈ {churn_prob:.0%})")

st.caption("Demo model from internship Sprint 4. Not for production decisions.")