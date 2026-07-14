AGE_GROUP_BINS = [0, 25, 35, 50, 100]
AGE_GROUP_LABELS = ["Young", "Adult", "Middle", "Senior"]

JOB_OPTIONS = [0, 1, 2, 3]
SEX_OPTIONS = ["male", "female"]
HOUSING_OPTIONS = ["own", "rent", "free"]
SAVING_OPTIONS = ["little", "moderate", "quite rich", "rich"]
CHECKING_OPTIONS = ["little", "moderate", "rich"]
PURPOSE_OPTIONS = [
    "car",
    "furniture/equipment",
    "radio/TV",
    "business",
    "education",
    "repairs",
    "vacation/others",
]

RISK_LABELS = {0: "LOW RISK", 1: "HIGH RISK"}
RISK_CLASS_NAMES = ["Low Risk", "High Risk"]
RISK_MAP = {"good": 0, "bad": 1, "0": 0, "1": 1}
SHAP_MAX_FEATURES = 10

PAGE_ICONS = {
    "home": "🏠",
    "dashboard": "📊",
    "prediction": "🤖",
    "shap": "📈",
    "performance": "📉",
}
