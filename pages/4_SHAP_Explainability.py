import streamlit as st

from credit_risk.constants import PAGE_ICONS
from credit_risk.explainability import (
    build_shap_explanation,
    plot_shap_summary,
    plot_shap_waterfall,
    summarize_top_features,
)
from credit_risk.utils import render_header


def main() -> None:
    st.set_page_config(page_title="SHAP Explainability", page_icon=PAGE_ICONS["shap"], layout="wide")
    render_header("📈 SHAP Explainability", "Understand model decisions with SHAP-driven explanations for the trained classifier.")

    try:
        _, transformed_df, feature_names, values, explanation, feature_importance = build_shap_explanation()
    except Exception as exc:
        st.error(f"Unable to load SHAP explainability: {exc}")
        st.stop()

    st.info("SHAP highlights which engineered and encoded features most strongly influence the predicted risk class.")

    st.subheader("Waterfall Plot")
    try:
        fig = plot_shap_waterfall(explanation)
        st.pyplot(fig)
    except Exception as exc:
        st.warning(f"Waterfall plot could not be rendered: {exc}")

    st.markdown("---")
    st.subheader("Summary Bar Plot")
    try:
        fig2 = plot_shap_summary(values, transformed_df, feature_names)
        st.pyplot(fig2)
    except Exception as exc:
        st.warning(f"Summary plot could not be rendered: {exc}")

    st.markdown("---")
    st.subheader("Top 10 Important Features")
    st.dataframe(feature_importance.rename("Mean Absolute SHAP Value"), use_container_width=True)

    st.markdown("---")
    st.subheader("Narrative Explanation")
    top_features = summarize_top_features(feature_importance)
    explanation_text = (
        "The customer is classified as HIGH RISK mainly because of "
        f"{top_features[0].lower()}, {top_features[1].lower()}, and {top_features[2].lower()}."
    )
    st.success(explanation_text)


if __name__ == "__main__":
    main()
