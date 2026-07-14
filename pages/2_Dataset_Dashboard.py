import streamlit as st

from credit_risk.dashboard import (
    boxplot_chart,
    build_dashboard_data,
    category_bar_chart,
    correlation_heatmap,
    histogram_chart,
    risk_distribution_chart,
)
from credit_risk.constants import PAGE_ICONS
from credit_risk.utils import render_header


def main() -> None:
    st.set_page_config(page_title="Dataset Dashboard", page_icon=PAGE_ICONS["dashboard"], layout="wide")
    render_header("📊 Dataset Dashboard", "Inspect the dataset, explore distributions, and understand risk patterns.")

    try:
        df = build_dashboard_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    total_customers = len(df)
    high_risk = int((df["Risk Label"] == "HIGH RISK").sum())
    low_risk = total_customers - high_risk
    default_rate = round(high_risk / total_customers * 100, 1) if total_customers else 0.0

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Customers", total_customers)
    metric_cols[1].metric("High Risk Customers", high_risk)
    metric_cols[2].metric("Low Risk Customers", low_risk)
    metric_cols[3].metric("Default Rate", f"{default_rate}%")

    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.altair_chart(risk_distribution_chart(df), use_container_width=True)
    with chart_col2:
        st.altair_chart(histogram_chart(df, "Age", "Age Distribution by Risk"), use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        st.altair_chart(histogram_chart(df, "Credit amount", "Credit Amount Distribution"), use_container_width=True)
    with chart_col4:
        st.altair_chart(histogram_chart(df, "Duration", "Loan Duration Distribution"), use_container_width=True)

    st.markdown("---")
    chart_col5, chart_col6 = st.columns(2)
    with chart_col5:
        st.altair_chart(category_bar_chart(df, "Housing", "Housing Distribution"), use_container_width=True)
    with chart_col6:
        st.altair_chart(category_bar_chart(df, "Purpose", "Purpose Distribution"), use_container_width=True)

    st.markdown("---")
    chart_col7, chart_col8 = st.columns(2)
    with chart_col7:
        st.altair_chart(category_bar_chart(df, "Saving accounts", "Saving Account Distribution"), use_container_width=True)
    with chart_col8:
        st.altair_chart(category_bar_chart(df, "Checking account", "Checking Account Distribution"), use_container_width=True)

    st.markdown("---")
    box_col1, box_col2 = st.columns(2)
    with box_col1:
        st.altair_chart(boxplot_chart(df, "Credit amount", "Credit Amount by Risk"), use_container_width=True)
    with box_col2:
        st.altair_chart(boxplot_chart(df, "Duration", "Duration by Risk"), use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation Heatmap")
    st.altair_chart(correlation_heatmap(df), use_container_width=True)

    st.markdown("---")
    st.subheader("Missing Values Summary")
    missing_summary = df.isna().sum().reset_index()
    missing_summary.columns = ["Column", "Missing Values"]
    missing_summary = missing_summary[missing_summary["Missing Values"] > 0]

    if missing_summary.empty:
        st.success("No missing values remain after preprocessing.")
    else:
        st.dataframe(missing_summary, use_container_width=True)

    st.markdown("---")
    st.subheader("Interactive Data Table")
    st.dataframe(df.head(200), use_container_width=True)


if __name__ == "__main__":
    main()
