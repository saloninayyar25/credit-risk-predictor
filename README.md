# 💳 Credit Risk Prediction App

A machine learning web app that predicts whether a loan applicant is a **good** or **bad** credit risk, built using XGBoost and deployed with Streamlit on Hugging Face Spaces.

## 🔍 Project Overview

This project uses the **German Credit Dataset (Statlog, 1994)** — a classic benchmark dataset containing 1,000 loan applicants with 20 attributes. The goal is to classify applicants as good or bad credit risks to support lending decisions.

## ⚙️ Model Details 

| Property           | Value                        |
| ------------------ | ---------------------------- |
| Algorithm          | XGBoost Classifier           |
| Training samples   | 1,000                        |
| Test samples       | 200 (80/20 split)            |
| Decision threshold | 0.35 (tuned for credit risk) |
| Bad class recall   | 0.35                         |
| Overall accuracy   | 70%                          |

**Features used:**

- Age, Job type, Housing type
- Saving accounts, Checking account balance
- Credit amount, Loan duration

## ⚖️ Fairness & Bias

The original dataset includes a `Sex` column. Analysis showed it was the **highest importance feature (14.7%)** — a sign of historical lending bias baked into 1994-era data, not genuine predictive signal.

**Decision: Sex was dropped from the model entirely.**

This reduced accuracy slightly but makes the model fairer and legally compliant with modern anti-discrimination lending laws.

The decision threshold was also lowered from the default 0.50 to **0.35** — in credit risk, missing a bad applicant (false negative) is more costly than wrongly flagging a good one (false positive).

## 🗂️ Dataset Issues Found

| Issue                             | Status                       | Fix Applied               |
| --------------------------------- | ---------------------------- | ------------------------- |
| Gender bias in Sex feature        | Confirmed (14.7% importance) | Dropped Sex column        |
| Class imbalance (70/30)           | Confirmed                    | `scale_pos_weight=2.33`   |
| Credit amount outliers            | Confirmed (max 18,424 DM)    | Capped at 99th percentile |
| Missing values in Saving/Checking | Confirmed (183 + 394 rows)   | Imputed with mode         |

## 🚀 How to Run Locally

```bash
git clone https://github.com/catcode94/credit-risk-predictor
cd credit-risk-predictor
pip install -r requirements.txt
streamlit run app2.py
```

## Deployment link: 
- https://saloninayyar-credit-risk-predictor.hf.space/

## 📦 Tech Stack used

- Python, XGBoost, scikit-learn, pandas
- Streamlit (frontend)
- Docker + Hugging Face Spaces (deployment)

## 📌 Limitations

- Dataset is from 1994 Germany — may not generalise to modern lending contexts
- Bad class recall of 0.35 means the model still misses a significant portion of bad applicants
- Not intended for production use — built as a portfolio/learning project 


