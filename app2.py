# Credit Risk Prediction App - Improved
# 1 = Good (lower risk), 0 = Bad (higher risk)

import streamlit as st
import pandas as pd
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide"
)

# ── Load model & encoders ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_credit_model.pkl")
    encoders = {
        col: joblib.load(f"{col}_encoder.pkl")
        for col in ["Sex", "Saving accounts", "Checking account"]
    }
    return model, encoders

model, encoders = load_artifacts()

# ── Sidebar: all inputs ────────────────────────────────────────────────────────
st.sidebar.header("📋 Applicant Information")
st.sidebar.markdown("Fill in the details below and click **Predict**.")

age = st.sidebar.number_input("Age", min_value=18, max_value=80, value=30)

sex = st.sidebar.selectbox("Sex", ["male", "female"])

job_labels = {
    0: "0 – Unskilled (non-resident)",
    1: "1 – Unskilled (resident)",
    2: "2 – Skilled",
    3: "3 – Highly skilled"
}
job_choice = st.sidebar.selectbox("Job Type", options=list(job_labels.keys()), format_func=lambda x: job_labels[x])

housing = st.sidebar.selectbox("Housing", ["own", "rent", "free"])

saving_accounts = st.sidebar.selectbox(
    "Saving Accounts",
    ["little", "moderate", "quite rich", "rich"]
)

checking_account = st.sidebar.selectbox(
    "Checking Account",
    ["little", "moderate", "rich"]
)

credit_amount = st.sidebar.number_input("Credit Amount (DM)", min_value=100, max_value=100000, value=5000, step=100)

duration = st.sidebar.number_input("Loan Duration (months)", min_value=1, max_value=72, value=12)

predict_btn = st.sidebar.button("🔍 Predict Credit Risk", use_container_width=True)

# ── Main area ──────────────────────────────────────────────────────────────────
st.title("💳 Credit Risk Prediction App")
st.markdown("Predict whether an applicant is a **Good** or **Bad** credit risk using an XGBoost model trained on German Credit Data.")

st.divider()

# Input summary
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Applicant Summary")
    summary_data = {
        "Field": ["Age", "Sex", "Job", "Housing", "Saving Accounts", "Checking Account", "Credit Amount", "Duration"],
        "Value": [
            age, sex, job_labels[job_choice], housing,
            saving_accounts, checking_account,
            f"DM {credit_amount:,}", f"{duration} months"
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    df_summary["Value"] = df_summary["Value"].astype(str)
    st.table(df_summary)

with col2:
    st.subheader("📊 Prediction Result")

    if predict_btn:
        # Build input dataframe
        input_df = pd.DataFrame({
            "Age": [age],
            "Sex": [encoders["Sex"].transform([sex])[0]],
            "Job": [job_choice],
            "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
            "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
            "Credit amount": [credit_amount],
            "Duration": [duration],
            "Housing_own": [1 if housing == "own" else 0],
            "Housing_rent": [1 if housing == "rent" else 0]
        })

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]  # [prob_bad, prob_good]
        good_prob = proba[1] * 100
        bad_prob = proba[0] * 100

        if prediction == 1:
            st.success("✅ **GOOD Credit Risk** — Lower risk applicant")
            st.markdown(f"**Confidence:** {good_prob:.1f}% probability of being a good credit risk")
        else:
            st.error("❌ **BAD Credit Risk** — Higher risk applicant")
            st.markdown(f"**Confidence:** {bad_prob:.1f}% probability of being a bad credit risk")

        # Probability bars
        st.markdown("#### Risk Probability Breakdown")
        st.markdown(f"🟢 Good Risk: **{good_prob:.1f}%**")
        st.progress(int(good_prob))
        st.markdown(f"🔴 Bad Risk: **{bad_prob:.1f}%**")
        st.progress(int(bad_prob))

        # Interpretation
        st.markdown("#### 📌 Interpretation")
        if good_prob >= 75:
            st.info("Strong approval candidate. Low likelihood of default.")
        elif good_prob >= 50:
            st.warning("Moderate risk. Consider additional checks or conditions.")
        else:
            st.error("High risk of default. Loan not recommended without collateral.")

    else:
        st.info("👈 Fill in the applicant details on the left and click **Predict Credit Risk**.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: XGBoost | Dataset: German Credit Data (Statlog) | Built with Streamlit")
