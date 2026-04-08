# 💳 Credit Risk Scoring System

**End-to-End Machine Learning Project | Production-Ready Architecture**

A professional ML engineering project that predicts customer credit risk using a complete pipeline-based architecture with preprocessing, model training, API deployment, and web UI integration.

---

## 🚀 Project Overview

Classifies customers into **High Risk** or **Low Risk** using machine learning on the German Credit Dataset (1000 samples).

**Flow:** `UI → API → ML Pipeline → Prediction → Risk Output`

---

## 🏗️ Tech Stack

| Layer           | Technology                           |
| --------------- | ------------------------------------ |
| Language        | Python 3.10                          |
| ML Framework    | Scikit-learn, XGBoost                |
| Pipeline        | sklearn Pipeline + ColumnTransformer |
| Backend         | Flask                                |
| Frontend        | Streamlit                            |
| Model Storage   | Joblib                               |
| Data Handling   | Pandas, NumPy                        |
| Visualization   | Matplotlib, Seaborn                  |

---

## 📁 Project Structure

```
Credit-Risk-Scoring-System/
│
├── backend/                        # Flask REST API
│   └── app.py
│
├── frontend/                       # Streamlit UI
│   └── app.py
│
├── src/                            # Reusable Python modules
│   ├── __init__.py
│   ├── preprocess.py               # Data loading, cleaning, feature engineering
│   ├── pipeline.py                 # Pipeline & model builder
│   └── evaluate.py                 # Evaluation utilities
│
├── data/
│   ├── raw/
│   │   └── german_credit_data.csv  # Original dataset
│   └── processed/                  # Cleaned data & train/test splits
│       ├── cleaned_data.csv
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── model/
│   ├── credit_risk_model.pkl       # Best production model
│   └── all_pipelines.pkl           # All trained pipelines
│
├── notebook/
│   ├── 01_data_preprocessing/
│   │   └── 01_data_preprocessing.ipynb
│   ├── 02_eda/
│   │   └── 02_eda.ipynb
│   ├── 03_statistical_analysis/
│   │   └── 03_statistical_analysis.ipynb
│   ├── 04_feature_engineering/
│   │   └── 04_feature_engineering.ipynb
│   ├── 05_model_training/
│   │   └── 05_model_training.ipynb
│   └── 06_model_evaluation_deployment/
│       └── 06_model_evaluation_deployment.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔄 ML Pipeline — Notebook Steps

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_data_preprocessing` | Load raw data, handle missing values, encode target, save cleaned CSV |
| 02 | `02_eda` | Distributions, box plots, default rates by category, pairplot |
| 03 | `03_statistical_analysis` | Correlation heatmap, Chi-Square test, Mann-Whitney U hypothesis test |
| 04 | `04_feature_engineering` | Create `Credit_per_Duration`, `Age_Group`, train-test split (80/20 stratified) |
| 05 | `05_model_training` | Build sklearn pipelines, 5-fold CV, train LR / RF / XGBoost |
| 06 | `06_model_evaluation_deployment` | Confusion matrices, ROC curves, feature importance, save best model |

> Run notebooks **in order** — each step reads outputs saved by the previous step.

---

## 📊 Models

- Logistic Regression
- Random Forest
- XGBoost

**Metrics:** Accuracy, ROC-AUC, Precision, Recall, F1-Score

---

## 🔌 API Endpoint

**POST** `/predict`

```json
// Input
{
  "Age": 35, "Job": 2, "Sex": "male", "Housing": "own",
  "Saving accounts": "little", "Checking account": "moderate",
  "Credit amount": 5000, "Duration": 24, "Purpose": "car"
}

// Output
{
  "prediction": 1,
  "risk_probability": 0.78,
  "risk_label": "HIGH RISK"
}
```

---

## ⚙️ How to Run

```bash
# 1. Create & activate environment
python -m venv ml_env
ml_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order (01 → 06)
jupyter notebook

# 4. Start backend
python backend/app.py

# 5. Start frontend
streamlit run frontend/app.py
```

---

## 👨💻 Author

**Anurag Patel** — B.Tech CSE (AI & ML)
