import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FinAI", page_icon="💳", layout="wide")
st.title("💳 FinAI – Credit Card Default Prediction")
st.write("Predict the likelihood of credit-card payment default next month.")

MODEL_PATH = os.path.join("models", "finai_model.pkl")
TARGET = "default payment next month"

FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4",
    "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

if not os.path.exists(MODEL_PATH):
    st.error("Model not found at models/finai_model.pkl. Run your training script first.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

model_features = getattr(model, "feature_names_in_", None)
if model_features is not None:
    model_features = list(model_features)
    if set(model_features) != set(FEATURES):
        st.error("Your saved model features do not match this app's 23 input fields.")
        st.write("Expected app features:", FEATURES)
        st.write("Model training features:", model_features)
        st.info("Retrain the model using cleaned_data_fixed.csv, with ID and target excluded from X.")
        st.stop()
else:
    model_features = FEATURES

st.subheader("Enter customer details")
with st.form("finai_form"):
    st.caption("Repayment status: use -2 to 8, following the dataset's coding.")
    a, b, c = st.columns(3)

    with a:
        limit_bal = st.number_input("Credit limit (LIMIT_BAL)", min_value=0, value=50000, step=10000)
        sex = st.selectbox("SEX", [1, 2])
        education = st.selectbox("EDUCATION", [0, 1, 2, 3, 4, 5, 6])
        marriage = st.selectbox("MARRIAGE", [0, 1, 2, 3])
        age = st.number_input("AGE", min_value=18, max_value=100, value=30)

    with b:
        pay_0 = st.selectbox("PAY_0", list(range(-2, 9)), index=4)
        pay_2 = st.selectbox("PAY_2", list(range(-2, 9)), index=4)
        pay_3 = st.selectbox("PAY_3", list(range(-2, 9)), index=4)
        pay_4 = st.selectbox("PAY_4", list(range(-2, 9)), index=4)
        pay_5 = st.selectbox("PAY_5", list(range(-2, 9)), index=4)
        pay_6 = st.selectbox("PAY_6", list(range(-2, 9)), index=4)

    with c:
        bill_amt1 = st.number_input("BILL_AMT1", value=30000)
        bill_amt2 = st.number_input("BILL_AMT2", value=28000)
        bill_amt3 = st.number_input("BILL_AMT3", value=25000)
        bill_amt4 = st.number_input("BILL_AMT4", value=22000)
        bill_amt5 = st.number_input("BILL_AMT5", value=20000)
        bill_amt6 = st.number_input("BILL_AMT6", value=18000)

    st.markdown("**Payment amounts**")
    p = st.columns(6)
    payment_defaults = [2000, 2000, 1500, 1500, 1000, 1000]
    payment_values = []
    for i, col in enumerate(p):
        with col:
            payment_values.append(
                st.number_input(f"PAY_AMT{i+1}", min_value=0, value=payment_defaults[i], step=500)
            )

    submitted = st.form_submit_button("Predict default risk")

if submitted:
    values = {
        "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education,
        "MARRIAGE": marriage, "AGE": age,
        "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3,
        "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
        "BILL_AMT1": bill_amt1, "BILL_AMT2": bill_amt2,
        "BILL_AMT3": bill_amt3, "BILL_AMT4": bill_amt4,
        "BILL_AMT5": bill_amt5, "BILL_AMT6": bill_amt6,
        "PAY_AMT1": payment_values[0], "PAY_AMT2": payment_values[1],
        "PAY_AMT3": payment_values[2], "PAY_AMT4": payment_values[3],
        "PAY_AMT5": payment_values[4], "PAY_AMT6": payment_values[5]
    }

    X_new = pd.DataFrame([[values[f] for f in model_features]], columns=model_features)
    try:
        pred = int(model.predict(X_new)[0])
        st.subheader("Prediction")
        if pred == 1:
            st.error("Higher estimated default risk: model predicts default.")
        else:
            st.success("Lower estimated default risk: model predicts no default.")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_new)[0]
            classes = list(model.classes_)
            if 1 in classes:
                probability = float(probs[classes.index(1)])
                st.metric("Estimated default probability", f"{probability:.1%}")
                st.progress(probability)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption("Experimental model output only. Not a credit decision or financial advice.")
