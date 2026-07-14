import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from credit_risk.constants import (
    CHECKING_OPTIONS,
    HOUSING_OPTIONS,
    JOB_OPTIONS,
    PURPOSE_OPTIONS,
    SAVING_OPTIONS,
    SEX_OPTIONS,
    PAGE_ICONS,
)
from credit_risk.predictor import build_prediction_frame, get_model, predict_customer
from credit_risk.utils import render_header


def main() -> None:
    st.set_page_config(page_title="Risk Prediction", page_icon=PAGE_ICONS["prediction"], layout="wide")
    render_header("🤖 Risk Prediction", "Enter customer information to get an instant credit risk score.")

    model = get_model()

    with st.form("prediction_form"):
        st.subheader("Customer Information")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", min_value=18, max_value=75, value=35)
            job = st.selectbox("Job", JOB_OPTIONS, index=2)
            sex = st.selectbox("Sex", SEX_OPTIONS, index=0)
            housing = st.selectbox("Housing", HOUSING_OPTIONS, index=0)
            saving = st.selectbox("Saving accounts", SAVING_OPTIONS, index=0)
        with col2:
            checking = st.selectbox("Checking account", CHECKING_OPTIONS, index=0)
            credit_amount = st.number_input("Credit Amount", min_value=0, value=5000, step=100)
            duration = st.number_input("Duration (Months)", min_value=1, value=24, step=1)
            purpose = st.selectbox("Purpose", PURPOSE_OPTIONS, index=0)

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        payload = {
            "age": age,
            "job": job,
            "credit_amount": credit_amount,
            "duration": duration,
            "sex": sex,
            "housing": housing,
            "saving": saving,
            "checking": checking,
            "purpose": purpose,
        }

        input_df = build_prediction_frame(payload)
        prediction, risk_probability, confidence, label = predict_customer(model, input_df)

        st.markdown("---")
        status_col, detail_col = st.columns([1.1, 1.2])
        with status_col:
            color = "#dc2626" if prediction == 1 else "#16a34a"
            st.markdown(
                f"""
                <div style='padding:1.2rem;border-radius:1rem;background:{color};color:white;text-align:center'>
                <h2 style='margin:0'>{label}</h2>
                <p style='margin:0.3rem 0 0 0'>Probability: {risk_probability:.2%}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(risk_probability if prediction == 1 else 1 - risk_probability)
            st.metric("Confidence", f"{confidence:.2%}")

        with detail_col:
            st.subheader("Engineered Features")
            engineered_df = input_df[["Credit_per_Duration", "Age_Group"]]
            st.dataframe(engineered_df, use_container_width=True)
            st.caption(f"Credit_per_Duration: {engineered_df['Credit_per_Duration'].iloc[0]:.2f}")
            st.caption(f"Age Group: {engineered_df['Age_Group'].iloc[0]}")

        st.markdown("---")
        st.subheader("Prediction Summary")
        summary_df = pd.DataFrame(
            [
                {
                    "Prediction": prediction,
                    "Risk Label": label,
                    "Probability": risk_probability,
                    "Confidence": confidence,
                }
            ]
        )
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("Probability Gauge")
        fig, ax = plt.subplots(figsize=(6, 2.2))
        ax.bar([0], [risk_probability], color=color, width=0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel("Probability")
        ax.set_title("High Risk Probability")
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
