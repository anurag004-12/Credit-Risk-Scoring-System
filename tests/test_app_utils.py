import pandas as pd
from credit_risk.feature_engineering import engineer_features
from credit_risk.predictor import build_prediction_frame


def test_engineer_features_creates_new_columns():
    df = pd.DataFrame({
        "Age": [30],
        "Credit amount": [1000],
        "Duration": [12],
    })

    result = engineer_features(df)

    assert "Credit_per_Duration" in result.columns
    assert "Age_Group" in result.columns
    assert result.loc[0, "Credit_per_Duration"] == 1000 / 12


def test_get_prediction_frame_has_expected_columns():
    payload = {
        "age": 35,
        "job": 1,
        "credit_amount": 2000,
        "duration": 24,
        "sex": "male",
        "housing": "own",
        "saving": "little",
        "checking": "moderate",
        "purpose": "car",
    }

    result = build_prediction_frame(payload)

    expected_columns = [
        "Age",
        "Job",
        "Credit amount",
        "Duration",
        "Sex",
        "Housing",
        "Saving accounts",
        "Checking account",
        "Purpose",
        "Credit_per_Duration",
        "Age_Group",
    ]

    assert list(result.columns) == expected_columns
