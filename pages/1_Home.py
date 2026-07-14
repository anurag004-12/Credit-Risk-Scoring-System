import streamlit as st

from credit_risk.constants import PAGE_ICONS
from credit_risk.utils import build_badge, render_header


def main() -> None:
    st.set_page_config(page_title="Home", page_icon=PAGE_ICONS["home"], layout="wide")
    render_header("🏠 Home", "A portfolio-ready credit risk scoring application built with Streamlit, XGBoost, and SHAP.")

    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0f172a,#1d4ed8);padding:1.5rem 1.5rem;border-radius:1rem;color:white;">
        <h3 style="margin:0 0 0.4rem 0;">Credit Risk Scoring System</h3>
        <p style="margin:0;">Predict whether a customer is low risk or high risk using a production-ready Random Forest pipeline.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Problem Statement")
    st.write(
        "Financial institutions need a dependable way to assess creditworthiness and flag high-risk applicants before extending loans."
    )

    st.markdown("### Business Objective")
    st.write(
        "This solution helps lenders reduce default exposure, improve decision transparency, and make more consistent credit-risk assessments."
    )

    st.markdown("### Workflow")
    workflow = ["Dataset", "EDA", "Feature Engineering", "Training", "Evaluation", "SHAP", "Prediction"]
    cols = st.columns(len(workflow))
    for col, step in zip(cols, workflow):
        col.markdown(
            f"<div style='text-align:center;padding:0.7rem;border:1px solid #dbeafe;border-radius:0.8rem;background:#eff6ff'><strong>{step}</strong></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Technologies")
    tech = ["Python", "Pandas", "NumPy", "Scikit-learn", "Random Forest", "SHAP", "Altair", "Matplotlib", "Streamlit", "Joblib"]
    st.markdown(" ".join([build_badge(item, "#2563eb") for item in tech]), unsafe_allow_html=True)

    st.markdown("### Model Information")
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Model", "Random Forest")
    info_col2.metric("Task", "Binary Classification")
    info_col3.metric("Target", "Low Risk / High Risk")

    with st.expander("Project Architecture", expanded=True):
        st.code(
            "Dataset → EDA → Feature Engineering → Training → Evaluation → SHAP → Prediction",
            language="text",
        )


if __name__ == "__main__":
    main()
