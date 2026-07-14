# 💳 Credit Risk Scoring System

A production-ready Streamlit application for predicting whether a customer is low risk or high risk using a trained XGBoost pipeline.

## Features

- Modern multi-page Streamlit UI for portfolio use
- Dataset dashboard with interactive Altair charts
- Risk prediction form with engineered features
- SHAP-based explainability for model decisions
- Model performance dashboard with metrics and visualisations

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```text
Credit-Risk-Scoring-System/
├── app.py
├── pages/
│   ├── home.py
│   ├── dashboard.py
│   ├── prediction.py
│   ├── shap_page.py
│   └── performance.py
├── model/
│   └── credit_risk_model.pkl
├── data/
│   └── raw/german_credit_data.csv
├── src/
│   └── app_utils.py
├── requirements.txt
├── setup.py
└── README.md
```

```powershell
# Activate your own virtual environment, not the committed environment folder.
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit opens at: `http://localhost:8501`

---

## Run tests

```bash
python -m pytest
```

## CI

A GitHub Actions workflow is included at `.github/workflows/ci.yml` to install dependencies and run the test suite on push and pull requests.

---

## Notes

- `ml_env/` is a local virtual environment and should not be committed to the repository.
- The app uses `src/app_utils.py` for shared loading and helper utilities.

---

## 👨💻 Author

**Anurag Patel** — B.Tech CSE (AI & ML)

---
