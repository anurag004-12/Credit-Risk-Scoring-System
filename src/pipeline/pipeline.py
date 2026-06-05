from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

NUM_FEATURES = ['Age', 'Job', 'Credit amount', 'Duration', 'Credit_per_Duration']
CAT_FEATURES = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Age_Group']


def build_preprocessor():
    return ColumnTransformer([
        ('num', StandardScaler(), NUM_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_FEATURES)
    ])


def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':        RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost':              XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }


def build_pipeline(model):
    return Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', model)
    ])
