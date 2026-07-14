from __future__ import annotations

import pandas as pd
import streamlit as st

from .config import DATA_PATH
from .preprocessing import clean_dataset
from .feature_engineering import engineer_features


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = clean_dataset(df)
    return engineer_features(df)


@st.cache_resource(show_spinner=False)
def load_model():
    import joblib

    from .config import MODEL_PATH

    return joblib.load(MODEL_PATH)
