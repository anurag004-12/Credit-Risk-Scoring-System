import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['Saving accounts', 'Checking account']:
        df[col].fillna(df[col].mode()[0], inplace=True)
    df['Risk'] = df['Risk'].map({'good': 0, 'bad': 1})
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Credit_per_Duration'] = df['Credit amount'] / df['Duration']
    df['Age_Group'] = pd.cut(
        df['Age'], bins=[0, 25, 35, 50, 100],
        labels=['Young', 'Adult', 'Middle', 'Senior']
    )
    return df
