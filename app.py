import streamlit as st

from credit_risk.constants import PAGE_ICONS
from credit_risk.utils import render_header


def main() -> None:
    st.set_page_config(
        page_title="Credit Risk Scoring System",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_header("Credit Risk Scoring System", "Navigate the app using the menu to review data, explainability, and predictions.")
    st.markdown(
        "Use the Streamlit page menu in the top-right corner to open Home, Dataset Dashboard, Risk Prediction, SHAP Explainability, or Model Performance."
    )


if __name__ == "__main__":
    main()
