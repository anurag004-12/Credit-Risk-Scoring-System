# 💳 Credit Risk Scoring System

**End-to-End Machine Learning Project | Production-Ready Architecture**

A professional ML engineering project that predicts customer credit risk using a complete pipeline-based architecture with preprocessing, model training, API deployment, and web UI integration.

---

## 🚀 Project Overview

This project builds a real-world **Credit Risk Scoring System** that classifies customers into **High Risk** or **Low Risk** categories using machine learning.

The system is designed as a full production pipeline:

**UI → API → ML Pipeline → Prediction → Risk Output**

---

## 🎯 Problem Statement

Financial institutions face high losses due to loan defaults. The goal is to build an intelligent system that:

* Predicts default risk
* Automates credit decision support
* Improves approval accuracy
* Reduces financial risk

---

## 🧠 Solution Architecture

```
User Input (Streamlit UI)
        ↓
Flask REST API
        ↓
ML Pipeline (Preprocessing + Model)
        ↓
Prediction Engine
        ↓
Risk Classification Output
```

---

## 🏗️ Tech Stack

| Layer           | Technology                           |
| --------------- | ------------------------------------ |
| Language        | Python                               |
| ML Framework    | Scikit-learn                         |
| Pipeline        | sklearn Pipeline + ColumnTransformer |
| Backend         | Flask                                |
| Frontend        | Streamlit                            |
| Model Storage   | Joblib                               |
| Data Handling   | Pandas, NumPy                        |
| Visualization   | Matplotlib, Seaborn                  |
| Version Control | Git + GitHub                         |

---

## 📁 Project Structure

```
Credit-Risk-Scoring-System/
│
├── backend/              # Flask API
│   └── app.py
│
├── frontend/             # Streamlit UI
│   └── app.py
│
├── model/                # Trained pipeline model
│   └── credit_risk_model.pkl
│
├── data/                 # Dataset
│
├── notebook/             # Jupyter notebooks
│   ├── 1_data_loading.ipynb
│   ├── 2_eda.ipynb
│   ├── 3_data_preprocessing.ipynb
│   ├── 4_feature_engineering.ipynb
│   ├── 5_model_training.ipynb
│   └── 6_pipeline_building.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔄 ML Pipeline Flow

1. Data Ingestion
2. Exploratory Data Analysis (EDA)
3. Data Cleaning
4. Feature Engineering
5. Model Training
6. Pipeline Construction
7. Model Evaluation
8. Model Serialization
9. API Deployment
10. UI Integration

---

## 📊 Models Implemented

* Logistic Regression
* Random Forest
* XGBoost

**Evaluation Metrics:**

* Accuracy
* ROC-AUC
* Recall (BAD class)
* Precision (BAD class)
* F1-Score

Final model selected using business-risk metrics.

---

## 🧱 ML Pipeline Design

```text
Raw Input → Preprocessing Pipeline → Feature Encoding → Scaling → Model → Prediction
```

**Benefits:**

* No manual encoding
* No feature mismatch
* API safe inputs
* Production reliability
* Deployment-ready model

---

## 🔌 API Endpoint

**POST** `/predict`

### Input JSON

```json
{
  "Age": 35,
  "Job": 2,
  "Saving accounts": "little",
  "Checking account": "moderate",
  "Credit amount": 5000,
  "Duration": 24,
  "Sex": "male",
  "Housing": "own",
  "Purpose": "car"
}
```

### Output JSON

```json
{
  "prediction": 1,
  "risk_probability": 0.78,
  "risk_label": "HIGH RISK"
}
```

---

## 🖥️ Frontend (Streamlit)

* User-friendly form
* Real-time prediction
* Risk classification display
* Probability score
* Interactive UI

---

## ⚙️ How to Run Project

### 1️⃣ Create Environment

```bash
python -m venv ml_env
ml_env\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train Model

Run notebooks in order from `notebook/` folder

### 4️⃣ Start Backend

```bash
python backend/app.py
```

### 5️⃣ Start Frontend

```bash
streamlit run frontend/app.py
```

---

## 🧪 Use Cases

* Loan approval automation
* Credit risk assessment
* Banking decision systems
* FinTech platforms
* Risk analytics
* Financial scoring engines

---

## 🧑‍💼 Resume Description

> Built a production-grade Credit Risk Scoring System using ML pipelines, Flask API, and Streamlit UI. Implemented preprocessing automation, feature engineering, model training, REST API integration, and end-to-end deployment-ready architecture for real-time credit risk prediction.

---

## 🚀 Future Enhancements

* Dockerization
* CI/CD pipeline
* Cloud deployment
* SHAP explainability
* Risk band classification
* Authentication
* Monitoring system
* Database integration
* Model versioning
* Drift detection

---

## 📜 License

This project is open-source and available for learning and educational purposes.

---

## 👨‍💻 Author

**Anurag Patel**
B.Tech CSE (AI & ML)
End-to-End ML Engineer Track

---

