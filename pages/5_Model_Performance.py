import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from credit_risk.constants import PAGE_ICONS
from credit_risk.evaluation import get_performance_metrics
from credit_risk.utils import render_header


def main() -> None:
    st.set_page_config(page_title="Model Performance", page_icon=PAGE_ICONS["performance"], layout="wide")
    render_header("📉 Model Performance", "Review the model’s evaluation metrics and quality indicators.")

    try:
        metrics = get_performance_metrics()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    metric_cols = st.columns(5)
    metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
    metric_cols[1].metric("Precision", f"{metrics['precision']:.4f}")
    metric_cols[2].metric("Recall", f"{metrics['recall']:.4f}")
    metric_cols[3].metric("F1 Score", f"{metrics['f1_score']:.4f}")
    metric_cols[4].metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
    st.caption("Active production model: Random Forest")

    st.markdown("---")
    conf_col, roc_col = st.columns(2)
    with conf_col:
        cm = pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actual Low Risk", "Actual High Risk"],
            columns=["Pred Low Risk", "Pred High Risk"],
        )
        st.subheader("Confusion Matrix")
        st.dataframe(cm, use_container_width=True)

    with roc_col:
        st.subheader("ROC Curve")
        try:
            from credit_risk.evaluation import get_roc_curve

            fpr, tpr, roc_auc = get_roc_curve()
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.plot(fpr, tpr, color="#1f77b4", label=f"ROC AUC = {roc_auc:.4f}")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend(loc="lower right")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"ROC curve could not be generated: {exc}")

    st.markdown("---")
    st.subheader("Classification Report")
    report_df = pd.DataFrame(metrics["classification_report"]).T
    st.dataframe(report_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Importance")
    try:
        from credit_risk.data_loader import load_model
        from credit_risk.evaluation import get_feature_importance

        model = load_model()
        importances = get_feature_importance(model, top_n=10)
        st.bar_chart(importances)
    except Exception as exc:
        st.warning(f"Feature importance plot could not be generated: {exc}")


if __name__ == "__main__":
    main()
