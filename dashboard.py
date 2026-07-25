import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("💳 Credit Card Fraud Detection Dashboard")
st.write("Upload transaction data (CSV) to flag potential fraud using a trained ML model.")

# Load the trained model
model = joblib.load("fraud_model.pkl")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Drop the actual label if it's present, so we don't accidentally use it as input
    if "Class" in data.columns:
        features = data.drop("Class", axis=1)
    else:
        features = data

    # Run predictions
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]  # confidence score for fraud class

    data["Predicted_Fraud"] = predictions
    data["Fraud_Probability"] = probabilities.round(3)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(data))
    col2.metric("Flagged as Fraud", int(data["Predicted_Fraud"].sum()))
    col3.metric("Fraud Rate", f"{(data['Predicted_Fraud'].mean() * 100):.2f}%")

    st.subheader("🚩 Flagged Transactions")
    st.dataframe(
        data[data["Predicted_Fraud"] == 1].sort_values("Fraud_Probability", ascending=False)
    )

    st.subheader("📋 All Transactions")
    st.dataframe(data)
else:
    st.info("👆 Upload a CSV file to get started. You can use a sample from your `data/creditcard.csv` file.")