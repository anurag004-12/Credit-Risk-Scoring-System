import streamlit as st
import requests

st.title("💳 Credit Risk Scoring System")

st.subheader("Customer Information")

# Example fields (adjust to your dataset columns)
age = st.number_input("Age", 18, 100)
income = st.number_input("Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_history = st.selectbox("Credit History", [0, 1])

if st.button("Predict Credit Risk"):
    input_data = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditHistory": credit_history
    }

    response = requests.post("http://127.0.0.1:5000/predict", json=input_data)

    if response.status_code == 200:
        result = response.json()

        if result["prediction"] == 1:
            st.error(f"🚨 HIGH RISK CUSTOMER\nRisk Probability: {result['risk_probability']:.2f}")
        else:
            st.success(f"✅ LOW RISK CUSTOMER\nRisk Probability: {result['risk_probability']:.2f}")
    else:
        st.error("API Error: Could not get prediction")