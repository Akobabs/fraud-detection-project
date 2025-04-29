import streamlit as st
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_and_preprocess_data
from model_training import train_and_evaluate

# Load and train model
model, explainer, metrics = train_and_evaluate()

st.title("E-Commerce Fraud Detection System")

# Display model performance
st.subheader("Model Performance")
st.write(metrics)

# Input form for transaction
st.subheader("Enter Transaction Details")
transaction_amt = st.number_input("Transaction Amount", 0.0, 10000.0)
product_cd = st.selectbox("Product Code", ['W', 'C', 'S', 'R', 'H'])
card1 = st.number_input("Card Identifier", 1000, 9999)
card4 = st.selectbox("Card Type", ['visa', 'mastercard', 'discover', 'amex'])
addr1 = st.number_input("Billing Region", 100, 999)
dist1 = st.number_input("Distance", 0.0, 1000.0)
device_type = st.selectbox("Device Type", ['mobile', 'desktop'])

# Predict and explain
if st.button("Predict"):
    # Create transaction DataFrame
    transaction = pd.DataFrame({
        'TransactionAmt': [transaction_amt],
        'ProductCD': [product_cd],
        'card1': [card1],
        'card4': [card4],
        'addr1': [addr1],
        'dist1': [dist1],
        'DeviceType': [device_type]
    })
    
    # Preprocess transaction (same as training)
    le = {col: {} for col in ['ProductCD', 'card4', 'DeviceType']}
    for col in ['ProductCD', 'card4', 'DeviceType']:
        le[col] = {'W': 0, 'C': 1, 'S': 2, 'R': 3, 'H': 4, 'visa': 0, 'mastercard': 1, 'discover': 2, 'amex': 3, 'mobile': 0, 'desktop': 1}
        transaction[col] = transaction[col].map(le[col])
    
    # Predict
    pred = model.predict(transaction)[0]
    st.write("Prediction:", "Fraud" if pred == 1 else "Legitimate")
    
    # SHAP explanation
    shap_values = explainer.shap_values(transaction)[1][0]
    st.write("Explanation (Feature Contributions):")
    for feature, value in zip(transaction.columns, shap_values):
        st.write(f"{feature}: {value:.4f}")
    
    # Display SHAP force plot
    st.subheader("SHAP Force Plot")
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value[1], shap_values, transaction)
    st_shap(force_plot)

# Helper function to display SHAP plots in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)