# 💳 Credit Risk Scoring System

An end-to-end credit risk scoring solution that predicts whether a customer is likely to be a low-risk or high-risk borrower. The project combines data preprocessing, feature engineering, model training, statistical validation, explainability, and a production-style Streamlit dashboard.

## 🌐 Live Demo

https://credit-risk-scoring-system-mjogvcpssqcqqmgr6kgzru.streamlit.app

---

## 🎯 Problem Definition

Credit risk assessment is critical for banks and financial institutions because poor credit decisions can lead to defaults, financial losses, and unstable lending portfolios. The goal of this system is to:

- classify applicants into risk categories,
- identify the most influential factors driving the prediction,
- provide transparent and interpretable outputs for business stakeholders.

---

## ✅ Solution Overview

This project builds a complete machine learning workflow for credit scoring:

- loads and cleans the German credit dataset,
- engineers meaningful features such as credit-per-duration and age groups,
- trains and evaluates multiple classifiers,
- selects the best-performing model for deployment,
- exposes the solution through an interactive Streamlit application.

The deployed app supports:

- dataset exploration and visualization,
- customer risk prediction,
- SHAP-based explainability,
- model performance monitoring with ROC curves and feature importance plots.

---

## 🔧 Methods Used

### 1. Data Preparation
- Data cleaning and preprocessing
- Handling missing values and categorical encoding
- Feature engineering for better predictive signal

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis
- Risk class analysis
- Relationship studies between features and default risk

### 3. Modeling
- Comparison of multiple classification models
- Best model selected as Random Forest for deployment
- Pipeline-based training for reproducibility

### 4. Evaluation Metrics
- Accuracy
- ROC-AUC
- Confusion matrix
- Classification report

### 5. Statistical Testing
- Mann–Whitney U test for comparing feature distributions between risk groups
- Used to validate whether certain predictors differ meaningfully across good and bad credit classes

### 6. Cross-Validation
- Cross-validation used to assess model robustness and generalization
- Helps reduce overfitting and supports model selection

### 7. Explainability
- SHAP values for local and global model interpretation
- Supports transparent business decision-making

---

## 📊 Project Structure

```text
Credit-Risk-Scoring-System/
├── app.py
├── pages/
├── src/
│   └── credit_risk/
├── model/
├── data/
├── notebook/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## ▶️ Run Locally

### Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Start the app

```powershell
streamlit run app.py
```

---

## 🧪 Run Tests

```bash
python -m pytest
```

---

## 🚀 Deployment Notes

This project is designed for Streamlit-based deployment and can be hosted on:

- Streamlit Community Cloud
- Azure App Service
- AWS EC2
- Docker container

---

## 👨‍💻 Author

**Anurag Patel** — B.Tech CSE (AI & ML)

---
