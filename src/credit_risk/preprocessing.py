from __future__ import annotations

from typing import Any

import pandas as pd

from .constants import RISK_MAP


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")], errors="ignore")

    for column in ["Saving accounts", "Checking account"]:
        if column in df.columns:
            mode_values = df[column].mode(dropna=True)
            fallback = mode_values[0] if not mode_values.empty else "little"
            df[column] = df[column].fillna(fallback)

    if "Risk" in df.columns:
        df["Risk"] = df["Risk"].astype(str).str.strip().str.lower()
        df["Risk"] = df["Risk"].map(RISK_MAP).astype("Int64")

    return df
