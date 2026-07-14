from __future__ import annotations

import altair as alt
import pandas as pd
from typing import cast

from .data_loader import load_dataset
from .constants import RISK_LABELS


def build_dashboard_data() -> pd.DataFrame:
    df = load_dataset()
    df = df.copy()
    df["Risk Label"] = df["Risk"].map(RISK_LABELS).astype(str)
    return df


def risk_distribution_chart(df: pd.DataFrame) -> alt.Chart:
    risk_counts = df["Risk Label"].value_counts().reset_index()
    risk_counts.columns = ["Risk Label", "Count"]

    return (
        alt.Chart(risk_counts)
        .mark_bar(size=30, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Risk Label:N", title="Risk Label"),
            y=alt.Y("Count:Q", title="Customers"),
            color=alt.Color("Risk Label:N", scale=alt.Scale(domain=list(RISK_LABELS.values()), range=["#10b981", "#ef4444"])),
            tooltip=["Risk Label", "Count"],
        )
        .properties(title="Risk Distribution")
    )


def histogram_chart(df: pd.DataFrame, field: str, title: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X(f"{field}:Q", bin=alt.Bin(maxbins=25), title=title),
            y=alt.Y("count()", title="Customers"),
            color=alt.Color("Risk Label:N", scale=alt.Scale(domain=list(RISK_LABELS.values()), range=["#10b981", "#ef4444"])),
        )
        .properties(title=title)
    )


def category_bar_chart(df: pd.DataFrame, field: str, title: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{field}:N", title=title),
            y=alt.Y("count()", title="Customers"),
            color=alt.Color(f"{field}:N"),
        )
        .properties(title=title)
    )


def boxplot_chart(df: pd.DataFrame, field: str, title: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_boxplot(size=50)
        .encode(
            x=alt.X("Risk Label:N", title="Risk Label"),
            y=alt.Y(f"{field}:Q", title=title),
            color=alt.Color("Risk Label:N", scale=alt.Scale(domain=list(RISK_LABELS.values()), range=["#10b981", "#ef4444"])),
        )
        .properties(title=title)
    )


def correlation_heatmap(df: pd.DataFrame) -> alt.Chart:
    correlation = df[["Age", "Job", "Credit amount", "Duration", "Credit_per_Duration"]].corr(numeric_only=True)
    return (
        alt.Chart(correlation.reset_index().melt("index"))
        .mark_rect()
        .encode(
            x=alt.X("variable:N", title="Feature"),
            y=alt.Y("index:N", title="Feature"),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="blueorange")),
            tooltip=["index", "variable", "value"],
        )
        .properties(title="Feature Correlation")
    )
