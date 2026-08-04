import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_credit_model.pkl")
    encoders = {
        "Saving accounts": joblib.load("Saving accounts_encoder.pkl"),
        "Checking account": joblib.load("Checking account_encoder.pkl")
    }
    return model, encoders

model, encoders = load_artifacts()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("📋 Applicant Information")
st.sidebar.markdown("Fill in the details below and click **Predict**.")

age = st.sidebar.number_input("Age", min_value=18, max_value=80, value=30)

job_labels = {
    0: "0 – Unskilled (non-resident)",
    1: "1 – Unskilled (resident)",
    2: "2 – Skilled",
    3: "3 – Highly skilled"
}
job_choice = st.sidebar.selectbox(
    "Job Type",
    options=list(job_labels.keys()),
    format_func=lambda x: job_labels[x]
)

housing = st.sidebar.selectbox("Housing", ["own", "rent", "free"])

saving_accounts = st.sidebar.selectbox(
    "Saving Accounts",
    ["little", "moderate", "quite rich", "rich"]
)

checking_account = st.sidebar.selectbox(
    "Checking Account",
    ["little", "moderate", "rich"]
)

credit_amount = st.sidebar.number_input(
    "Credit Amount (DM)", min_value=100, max_value=18500, value=5000, step=100
)

duration = st.sidebar.number_input(
    "Loan Duration (months)", min_value=1, max_value=72, value=12
)

predict_btn = st.sidebar.button("🔍 Predict Credit Risk", use_container_width=True)

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("💳 Credit Risk Prediction App")
st.markdown(
    "Predicts whether an applicant is a **Good** or **Bad** credit risk "
    "using an XGBoost model trained on the German Credit Dataset."
)
st.info(
    "⚠️ **Fairness note:** The Sex feature was removed from this model due to "
    "documented gender bias in the original 1994 dataset. Predictions are based "
    "solely on financial and demographic factors.",
    icon="ℹ️"
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Applicant Summary")
    summary_data = {
        "Field": ["Age", "Job", "Housing", "Saving Accounts",
                  "Checking Account", "Credit Amount", "Duration"],
        "Value": [
            str(age),
            job_labels[job_choice],
            housing,
            saving_accounts,
            checking_account,
            f"DM {credit_amount:,}",
            f"{duration} months"
        ]
    }
    st.table(pd.DataFrame(summary_data))

with col2:
    st.subheader("📊 Prediction Result")

    if predict_btn:
        # Cap credit amount same as training
        credit_amount_capped = min(credit_amount, 18500 * 0.99)

        input_df = pd.DataFrame({
            "Age": [age],
            "Job": [job_choice],
            "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
            "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
            "Credit amount": [credit_amount_capped],
            "Duration": [duration],
            "Housing_own": [1 if housing == "own" else 0],
            "Housing_rent": [1 if housing == "rent" else 0]
        })

        # Apply same 0.35 threshold used during training
        y_proba = model.predict_proba(input_df)
        bad_proba = y_proba[:, 0][0]
        good_proba = y_proba[:, 1][0]
        prediction = 0 if bad_proba > 0.35 else 1

        good_pct = round(good_proba * 100, 1)
        bad_pct = round(bad_proba * 100, 1)

        if prediction == 1:
            st.success("✅ **GOOD Credit Risk** — Lower risk applicant")
            st.markdown(f"**Confidence:** {good_pct}% probability of good credit risk")
        else:
            st.error("❌ **BAD Credit Risk** — Higher risk applicant")
            st.markdown(f"**Confidence:** {bad_pct}% probability of bad credit risk")

        st.markdown("#### Risk Probability Breakdown")
        st.markdown(f"🟢 Good Risk: **{good_pct}%**")
        st.progress(int(good_pct))
        st.markdown(f"🔴 Bad Risk: **{bad_pct}%**")
        st.progress(int(bad_pct))

        st.markdown("#### 📌 Interpretation")
        if bad_proba <= 0.35:
            if good_pct >= 75:
                st.info("Strong approval candidate. Low likelihood of default.")
            else:
                st.warning("Moderate-good profile. Standard checks recommended.")
        elif bad_proba <= 0.55:
            st.warning("Borderline risk. Consider additional checks or collateral.")
        else:
            st.error("High risk of default. Loan not recommended without collateral.")

    else:
        st.info("👈 Fill in the applicant details on the left and click **Predict Credit Risk**.")

st.divider()
st.caption("Model: XGBoost | Dataset: German Credit Data (Statlog, 1994) | Built with Streamlit")
