from __future__ import annotations

import json
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import PERFORMANCE_PATH, TEST_X_PATH, TEST_Y_PATH
from .data_loader import load_model
from .constants import RISK_CLASS_NAMES


@st.cache_data(show_spinner=False)
def get_performance_metrics() -> dict[str, object]:
    if PERFORMANCE_PATH.exists():
        with PERFORMANCE_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    X_test = pd.read_csv(TEST_X_PATH, index_col=0)
    y_test = pd.read_csv(TEST_Y_PATH, index_col=0).squeeze()

    model = load_model()
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, target_names=RISK_CLASS_NAMES),
    }

    with PERFORMANCE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def get_roc_curve() -> tuple[list[float], list[float], float]:
    """Compute ROC curve values for the test set using the installed model."""
    X_test = pd.read_csv(TEST_X_PATH, index_col=0)
    y_test = pd.read_csv(TEST_Y_PATH, index_col=0).squeeze()
    model = load_model()

    # predict_proba may not exist for all estimators; handle gracefully
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except Exception:
        # fallback: use decision_function if available
        try:
            probs = model.decision_function(X_test)
        except Exception:
            probs = None

    if probs is None:
        raise RuntimeError("Model does not provide probabilities or decision function for ROC computation.")

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = float(auc(fpr, tpr))
    return fpr.tolist(), tpr.tolist(), roc_auc


def get_feature_importance(model, top_n: int = 10):
    """Return a pandas Series of top feature importances with names aligned to preprocessor output."""
    import pandas as pd

    feature_names = None
    try:
        pre = model.named_steps.get("preprocessor")
        if pre is not None:
            feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = getattr(model, "feature_names_in_", None)

    importances = None
    try:
        importances = model.named_steps["model"].feature_importances_
    except Exception:
        # fallback if model is not a pipeline or uses different attribute name
        importances = getattr(model, "feature_importances_", None)

    if importances is None:
        raise RuntimeError("Model does not expose feature importances.")

    if feature_names is None:
        # create generic names
        feature_names = [f"f_{i}" for i in range(len(importances))]

    series = pd.Series(importances, index=feature_names)

    def _prettify(name: str, max_len: int = 28) -> str:
        s = str(name)
        # Remove common pipeline prefixes
        for p in ("cat__", "num__", "preprocessor__", "onehot__", "ohe__"):
            if s.startswith(p):
                s = s[len(p) :]
                break

        # Replace common separators with a readable delimiter
        s = s.replace("__", " ").replace("=", " ").replace("-", " ")
        s = s.replace("_", " ")
        s = s.strip()

        # Collapse repeated whitespace
        s = " ".join(s.split())

        # Truncate long names for compact display
        if len(s) > max_len:
            s = s[: max_len - 1].rstrip() + "…"
        return s

    pretty_names = [_prettify(n) for n in series.index.tolist()]

    # Ensure unique names by appending counts when collisions occur
    seen: dict[str, int] = {}
    unique_names: list[str] = []
    for nm in pretty_names:
        if nm in seen:
            seen[nm] += 1
            unique_names.append(f"{nm} ({seen[nm]})")
        else:
            seen[nm] = 0
            unique_names.append(nm)

    series.index = unique_names
    return series.sort_values(ascending=False).head(top_n)
