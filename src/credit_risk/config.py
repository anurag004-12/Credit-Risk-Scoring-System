from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "raw" / "german_credit_data.csv"
MODEL_PATH = ROOT_DIR / "model" / "credit_risk_model.pkl"
PERFORMANCE_PATH = ROOT_DIR / "model" / "performance_metrics.json"
TEST_X_PATH = ROOT_DIR / "data" / "processed" / "X_test.csv"
TEST_Y_PATH = ROOT_DIR / "data" / "processed" / "y_test.csv"
RANDOM_SEED = 42
SHAP_MAX_FEATURES = 10
