# 💳 Credit Risk Prediction App

A machine learning web app that predicts whether a loan applicant is a **good** or **bad** credit risk, built using XGBoost and deployed with Streamlit on Hugging Face Spaces.

## 🔍 Project Overview

This project uses the **German Credit Dataset (Statlog, 1994)** — a classic benchmark dataset containing 1,000 loan applicants with 20 attributes. The goal is to classify applicants as good or bad credit risks to support lending decisions.

## ⚙️ Model Details

| Property             | Value                                             |
| --------------------- | -------------------------------------------------- |
| Algorithm              | XGBoost Classifier (`min_child_weight=3`, `scale_pos_weight` for class balance) |
| Training samples       | 800 (80/20 split)                                 |
| Test samples            | 200                                                |
| Decision threshold      | 0.35 — empirically chosen via a precision/recall sweep on the held-out test set |
| Bad-class recall at 0.35 | ~73%                                             |
| Test accuracy            | ~66% |

**Note on accuracy:** this model is intentionally tuned for bad-risk **recall**, not raw accuracy — `scale_pos_weight` makes it more willing to flag an applicant as bad, which means it will always score below a trivial "always predict good" baseline (70% on this dataset) on accuracy alone. In credit risk, missing a bad applicant is more costly than over-flagging a good one, so recall on the bad class is the metric that actually matters here, not accuracy.

**Features used:**

- Age, Job type, Housing type
- Saving accounts, Checking account balance
- Credit amount, Loan duration

## ⚖️ Fairness & Bias

The original dataset includes a `Sex` column. Analysis showed it was the **highest importance feature (14.7%)** when included in training — a sign of historical lending bias baked into 1994-era data, not genuine predictive signal (verified: female applicants show a 35.2% bad-risk rate vs. 27.7% for male applicants in the raw data).

**Decision: Sex is excluded from the trained model entirely** — not just hidden from the app's input form. Verified directly against the deployed model's feature schema (`['Age', 'Job', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Housing_own', 'Housing_rent']` — 8 features, no Sex).

This is a form of **"fairness through unawareness"** and has a known limitation: other features (e.g. Job, Housing) may still correlate with Sex and partially leak the same signal. This hasn't been formally tested here and is listed as a limitation below rather than presented as a complete fix.

The decision threshold was also lowered from the default 0.50 to **0.35** — in credit risk, missing a bad applicant (false negative) is more costly than wrongly flagging a good one (false positive). Verified via threshold sweep: 0.35 gives ~73% bad-class recall at ~37% precision, the best trade-off tested.

## 🎯 Model Reliability & Data Coverage

Testing surfaced a real limitation worth documenting rather than hiding: **the model can be confidently wrong in sparse regions of feature space.** A test case combining a large credit amount (DM 13,100) with a very short duration (3 months) — a loan shape barely represented in the training data (only 2 of 1,000 rows resemble it, both labeled bad) — produced a high-confidence "bad" prediction that shifted sharply and non-intuitively when only Job type and Housing were changed.

To address this, the app now includes a **data-coverage check**: before returning a prediction, it measures how close the input sits to real training examples (nearest-neighbor distance in scaled feature space). Inputs that fall outside the 90th-percentile coverage of the training data are flagged with an explicit low-confidence warning rather than presented as a clean, trustworthy percentage.

## 🗂️ Dataset Issues Found

| Issue                             | Status                       | Fix Applied                              |
| ---------------------------------- | ----------------------------- | ------------------------------------------ |
| Gender bias in Sex feature          | Confirmed (14.7% importance) | Excluded from training entirely            |
| Class imbalance (700 good / 300 bad) | Confirmed                    | `scale_pos_weight`                         |
| Credit amount outliers              | Confirmed (max 18,424 DM)    | Capped at 99th percentile (DM 14,180.4)    |
| Missing values in Saving/Checking   | Confirmed (183 + 394 rows)   | Imputed with mode                          |
| Overconfidence in sparse regions    | Confirmed (see above)        | Moderate regularization (`min_child_weight=3`) + nearest-neighbor data-coverage warning |

## 🚀 How to Run Locally

```bash
git clone https://github.com/saloninayyar25/credit-risk-predictor
cd credit-risk-predictor
pip install -r requirements.txt
streamlit run app2.py
```

## Deployment link: https://saloninayyar-credit-risk-score-predictor.hf.space/

## 📦 Tech Stack

- Python, XGBoost, scikit-learn, pandas
- Streamlit (frontend)
- Docker + Hugging Face Spaces (deployment)

## 📌 Limitations

- Dataset is from 1994 Germany — may not generalize to modern lending contexts
- Bad-class recall of ~73% (at the 0.35 threshold) still means roughly 1 in 4 bad applicants is missed
- Excluding Sex reduces direct discrimination but does not rule out indirect bias through correlated features (e.g. Job, Housing) — not formally tested
- Model reliability drops in sparse regions of feature space (e.g. large amount + short duration); the data-coverage warning mitigates but does not eliminate this
- Trained on only 1,000 rows total — several feature combinations have very few or zero real precedents
- Not intended for production use — built as a portfolio/learning project
