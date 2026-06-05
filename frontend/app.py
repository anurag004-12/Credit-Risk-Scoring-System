import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "german_credit_data.csv"

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")], errors="ignore")
    df["Risk Label"] = df["Risk"].map({"good": "LOW RISK", "bad": "HIGH RISK"})
    return df


def show_dashboard(df: pd.DataFrame) -> None:
    st.title("📊 Credit Risk Dashboard")
    st.markdown("Use this dashboard to explore the dataset and compare risk segments.")

    total = len(df)
    high_risk = int((df["Risk Label"] == "HIGH RISK").sum())
    low_risk = total - high_risk
    high_risk_pct = round(high_risk / total * 100, 1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total records", total)
    col2.metric("High risk", f"{high_risk}", f"{high_risk_pct}%")
    col3.metric("Low risk", low_risk)

    st.markdown("---")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        risk_count = (
            df["Risk Label"]
            .value_counts()
            .reset_index(name="Count")
            .rename(columns={"index": "Risk Label"})
        )
        chart = alt.Chart(risk_count).mark_bar().encode(
            x=alt.X("Risk Label:N", title="Risk"),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color("Risk Label:N", legend=None)
        ).properties(title="Risk Distribution")
        st.altair_chart(chart, use_container_width=True)

    with chart_cols[1]:
        age_hist = alt.Chart(df).mark_bar().encode(
            x=alt.X("Age:Q", bin=alt.Bin(maxbins=20), title="Age"),
            y=alt.Y("count()", title="Number of customers"),
            color=alt.Color("Risk Label:N", scale=alt.Scale(scheme="tableau10"))
        ).properties(title="Age Distribution by Risk")
        st.altair_chart(age_hist, use_container_width=True)

    st.markdown("---")
    breakdown_cols = st.columns(2)
    with breakdown_cols[0]:
        credit_box = alt.Chart(df).mark_boxplot().encode(
            x=alt.X("Risk Label:N", title="Risk"),
            y=alt.Y("Credit amount:Q", title="Credit Amount"),
            color=alt.Color("Risk Label:N", legend=None)
        ).properties(title="Credit Amount by Risk")
        st.altair_chart(credit_box, use_container_width=True)

    with breakdown_cols[1]:
        purpose_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("Purpose:N", sort="-y", title="Purpose"),
            y=alt.Y("count()", title="Count"),
            color=alt.Color("Risk Label:N", scale=alt.Scale(scheme="tableau10"))
        ).properties(title="Credit Purpose by Risk")
        st.altair_chart(purpose_chart, use_container_width=True)

    st.markdown("---")
    st.subheader("Data sample")
    st.dataframe(df.head(10), use_container_width=True)


def show_prediction_form() -> None:
    st.title("💳 Credit Risk Scoring System")
    st.subheader("Enter customer details")

    age = st.slider("Age", 18, 75)
    job = st.selectbox("Job (0–3)", [0, 1, 2, 3])
    credit_amount = st.number_input("Credit Amount", min_value=0)
    duration = st.number_input("Duration (months)", min_value=1)

    sex = st.selectbox("Sex", ["male", "female"])
    housing = st.selectbox("Housing", ["own", "rent", "free"])
    saving = st.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich"])
    checking = st.selectbox("Checking account", ["little", "moderate", "rich"])
    purpose = st.selectbox(
        "Purpose",
        [
            "car", "furniture/equipment", "radio/TV", "business",
            "education", "repairs", "vacation/others"
        ]
    )

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
            API_URL = os.environ.get("API_URL", "http://127.0.0.1:5000")
            response = requests.post(f"{API_URL}/predict", json=input_data)
            result = response.json()

            if "error" in result:
                st.error(result["error"])
            else:
                if result["prediction"] == 1:
                    st.error(f"HIGH RISK\nProbability: {result['risk_probability']}")
                else:
                    st.success(f"LOW RISK\nProbability: {result['risk_probability']}")

        except requests.RequestException:
            st.error("Flask server not running!")


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose view", ["Prediction", "Dashboard"])

    df = load_data()

    if page == "Dashboard":
        show_dashboard(df)
    else:
        show_prediction_form()


if __name__ == "__main__":
    main()
