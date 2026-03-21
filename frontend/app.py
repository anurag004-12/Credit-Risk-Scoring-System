import streamlit as st
import requests

st.set_page_config(page_title="Credit Risk App")

st.title("💳 Credit Risk Scoring System")

st.subheader("Enter Customer Details")

# 🔹 Inputs (MUST match training features)
age = st.slider("Age", 18, 75)
job = st.selectbox("Job (0–3)", [0, 1, 2, 3])
credit_amount = st.number_input("Credit Amount", min_value=0)
duration = st.number_input("Duration (months)", min_value=1)

sex = st.selectbox("Sex", ["male", "female"])
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving = st.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich"])
checking = st.selectbox("Checking account", ["little", "moderate", "rich"])
purpose = st.selectbox("Purpose", [
    "car", "furniture/equipment", "radio/TV", "business", "education", "repairs", "vacation/others"
])

# 🔹 Send request
if st.button("Predict Risk"):

    input_data = {
        "Age": age,
        "Job": job,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Sex": sex,
        "Housing": housing,
        "Saving accounts": saving,
        "Checking account": checking,
        "Purpose": purpose
    }

    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json=input_data
        )

        result = response.json()

        if "error" in result:
            st.error(result["error"])
        else:
            if result["prediction"] == 1:
                st.error(f"HIGH RISK\nProbability: {result['risk_probability']}")
            else:
                st.success(f"LOW RISK\nProbability: {result['risk_probability']}")

    except:
        st.error("Flask server not running!")