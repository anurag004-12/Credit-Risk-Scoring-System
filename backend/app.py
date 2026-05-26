from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import engineer_features

app = Flask(__name__)

# Load full pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "credit_risk_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return "Credit Risk API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Convert to DataFrame
        df = pd.DataFrame([data])

        #  Feature engineering (same as training)
        df = engineer_features(df)

        # Prediction (pipeline handles everything)
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "risk_probability": round(float(probability), 3),
            "risk_label": "HIGH RISK" if prediction == 1 else "LOW RISK"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, use_reloader=False)