import streamlit as st
import pandas as pd
import joblib
import sys
import os

# Import preprocessing from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import preprocess_uploaded_data

def display():
    st.title("📊 Epilepsy Detection Results")

    uploaded_file = st.file_uploader("Upload EEG CSV file", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("🧾 Uploaded Data Preview", data.head())

            # Preprocess and predict
            X = preprocess_uploaded_data(data)
            model = joblib.load("Epilepcy/models/ensemble_model.pkl")
            prediction = model.predict(X)

            result = "🟥 Seizure Detected" if prediction[0] == 1 else "🟩 No Seizure Detected"
            st.success(f"Prediction: {result}")

        except Exception as e:
            st.error(f"❌ Error processing the file: {e}")
