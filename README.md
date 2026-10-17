# Sprint 4 — Telco Churn Live Demo

Streamlit app that loads a trained scikit-learn pipeline and predicts customer churn.

## Live demo
https://sprint4-telco-churn-demo-encmqvxfcwc74qjara2m8o.streamlit.app/

## What’s included
- app.py I
- churn_model.joblib
- requirements.txt

## How to use the demo
1. Open the live link
2. Set customer fields (contract, tenure, internet, charges, etc.)
3. Click **Predict churn**
4. Read Yes/No and approximate churn probability

## How it was built
- Model trained on IBM Telco Customer Churn data 
- Features include engineered columns: tenure_years, num_services, long_contract, avg_charge_per_tenure
- Serialized with joblib
- Deployed on Streamlit

## Note
This is only a demo
