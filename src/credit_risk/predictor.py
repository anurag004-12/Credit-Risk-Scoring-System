from __future__ import annotations

import pandas as pd
import streamlit as st

from .data_loader import load_model


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


def build_prediction_frame(payload: dict[str, object]) -> pd.DataFrame:
    record = {
        "Age": payload["age"],
        "Job": payload["job"],
        "Credit amount": payload["credit_amount"],
        "Duration": payload["duration"],
        "Sex": payload["sex"],
        "Housing": payload["housing"],
        "Saving accounts": payload["saving"],
        "Checking account": payload["checking"],
        "Purpose": payload["purpose"],
    }
    df = pd.DataFrame([record])
    # Apply the same feature engineering used in training
    from .feature_engineering import engineer_features

    df = engineer_features(df)
    # ensure Age_Group is a string for display/UI consistency
    if "Age_Group" in df.columns:
        df["Age_Group"] = df["Age_Group"].astype(str)
    return df


def predict_customer(model, input_df: pd.DataFrame) -> tuple[int, float, float, str]:
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    risk_probability = float(probabilities[1])
    confidence = float(abs(risk_probability - 0.5) * 2)
    label = "HIGH RISK" if prediction == 1 else "LOW RISK"
    return prediction, risk_probability, confidence, label
