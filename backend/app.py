from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "credit_risk_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "risk_probability": float(probability),
        "risk_label": "HIGH RISK" if prediction == 1 else "LOW RISK"
    })

if __name__ == "__main__":
    app.run(debug=True)