from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .data_loader import load_dataset, load_model
from .constants import RISK_MAP, RISK_LABELS, SHAP_MAX_FEATURES


def get_shap_explainer():
    model = load_model()
    explainer = shap.TreeExplainer(model.named_steps["model"])
    return explainer


def build_shap_explanation():
    df = load_dataset()
    X = df.drop(columns=["Risk"], errors="ignore")
    model = load_model()
    preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out().tolist()

# Clean feature names for display
    feature_names = [
        name.replace("num__", "")
        .replace("cat__", "")
        .replace("_", " ")
        .title()
        for name in feature_names
    ]
    explainer = get_shap_explainer()
    shap_values_raw = explainer.shap_values(transformed)



    values = np.asarray(shap_values_raw)

# SHAP >= 0.46 returns (samples, features, classes)
    if values.ndim == 3:
        values = values[:, :, 1]          # Keep only HIGH RISK class
        base_value = explainer.expected_value[1]
    else:
        base_value = explainer.expected_value

    print("Values shape:", values.shape)
    print("Values ndim:", values.ndim)

# If SHAP returned 3 dimensions, print their meaning
    if values.ndim == 3:
        print("Dimension 0 (samples):", values.shape[0])
        print("Dimension 1:", values.shape[1])
        print("Dimension 2:", values.shape[2])

    explanation = shap.Explanation(
        values=values,
        base_values=np.repeat(base_value, values.shape[0]),
        data=transformed,
        feature_names=feature_names,
    )

    feature_importance = np.abs(values).mean(axis=0)
    feature_importance = pd.Series(feature_importance, index=feature_names).sort_values(ascending=False).head(SHAP_MAX_FEATURES)

    return X, transformed, feature_names, values, explanation, feature_importance


def plot_shap_waterfall(explanation):
    import matplotlib.pyplot as plt
    # Normalize multi-output explanations to a single-output per-sample Explanation
    import numpy as _np

    def _select_output(expl: shap.Explanation, output_index: int | None = None) -> shap.Explanation:
        vals = _np.asarray(expl.values)
        # vals shape possibilities: (n_samples, n_features) or (n_samples, n_outputs, n_features)
        if vals.ndim == 3:
            # choose output_index (default last class) if present
            out_idx = output_index if output_index is not None else vals.shape[1] - 1
            if out_idx >= vals.shape[1]:
                out_idx = 0
            sel_vals = vals[:, out_idx, :]
            base = expl.base_values
            base_arr = _np.asarray(base)
            base_sel = base_arr[out_idx] if base_arr.ndim > 0 and base_arr.shape[0] > out_idx else float(base_arr)
            return shap.Explanation(values=sel_vals, base_values=base_sel, data=expl.data, feature_names=expl.feature_names)
        return expl

    plt.close("all")
    expl = _select_output(explanation)
    # waterfall expects a single-sample explanation with 1D values.
    vals = _np.asarray(expl.values)
    if vals.ndim == 2:
        single_vals = vals[0]
        # base value selection
        base = expl.base_values
        base_arr = _np.asarray(base)
        base_val = base_arr[0] if base_arr.ndim > 0 and base_arr.shape[0] > 0 else float(base_arr)
        # data for the single sample
        data_single = expl.data[0] if expl.data is not None else None
        single = shap.Explanation(values=single_vals, base_values=base_val, data=data_single, feature_names=expl.feature_names)
    else:
        try:
            single = expl[0]
        except Exception:
            single = expl

    shap.plots.waterfall(single, max_display=SHAP_MAX_FEATURES, show=False)
    fig = plt.gcf()
    return fig


def plot_shap_summary(values, transformed, feature_names):
    import matplotlib.pyplot as plt

    plt.close("all")

    shap.summary_plot(
        values,
        transformed,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )

    fig = plt.gcf()
    return fig


def summarize_top_features(feature_importance):
    return feature_importance.index.tolist()[:3]
