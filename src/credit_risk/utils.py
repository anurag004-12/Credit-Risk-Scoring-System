from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st


def render_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)
    st.markdown("---")


def build_badge(label: str, color: str) -> str:
    return f"<span style='background:{color};color:white;padding:0.2rem 0.5rem;border-radius:999px;font-size:0.8rem'>{label}</span>"


def dataframe_summary(df: pd.DataFrame, sample_size: int = 200) -> pd.DataFrame:
    return df.head(sample_size)


def numeric_metrics(columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column != "Risk"]
