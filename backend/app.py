from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from src.components.preprocess import engineer_features
from src.logger.logger import get_logger

app = Flask(__name__)

logger = get_logger(
    __name__,
    log_file_path=os.path.join(os.path.dirname(__file__), "..", "logs", "backend.log")
)

# Load full pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "credit_risk_model.pkl")
model = joblib.load(MODEL_PATH)
logger.info("Loaded credit risk model from %s", MODEL_PATH)

@app.route("/")
def home():
    return "Credit Risk API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        logger.info("Received prediction request: %s", data)

        # Convert to DataFrame
        df = pd.DataFrame([data])

        #  Feature engineering (same as training)
        df = engineer_features(df)

        # Prediction (pipeline handles everything)
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        result = {
            "prediction": int(prediction),
            "risk_probability": round(float(probability), 3),
            "risk_label": "HIGH RISK" if prediction == 1 else "LOW RISK"
        }
        logger.info("Prediction result: %s", result)
        return jsonify(result)

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, use_reloader=False)