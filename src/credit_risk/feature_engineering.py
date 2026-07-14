from __future__ import annotations

import pandas as pd

from .constants import AGE_GROUP_BINS, AGE_GROUP_LABELS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Credit_per_Duration"] = df["Credit amount"] / df["Duration"]
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=AGE_GROUP_BINS,
        labels=AGE_GROUP_LABELS,
        include_lowest=True,
    )
    return df
