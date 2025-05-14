import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath):
    """
    Load EEG data and preprocess it.
    Removes unnamed columns, converts multi-class to binary (seizure vs non-seizure),
    and scales the data.
    """
    df = pd.read_csv(filepath)

    # Drop unnamed or index columns if present
    df = df.drop(columns=['Unnamed', 'Unnamed: 0'], errors='ignore')

    # Convert multi-class to binary classification: 1 (seizure) vs 0 (non-seizure)
    df['y'] = df['y'].apply(lambda x: 1 if x == 1 else 0)

    X = df.drop('y', axis=1)
    y = df['y']

    # Load and apply scaler saved during training
    scaler = joblib.load("models/scaler.pkl")
    X_scaled = scaler.transform(X)

    return X_scaled, y

def preprocess_uploaded_data(df):
    """
    Preprocess the uploaded EEG data before prediction.
    Aligns the features with the trained model and applies saved scaler.
    """

    # Drop unnamed/index columns
    df = df.drop(columns=['Unnamed', 'Unnamed: 0'], errors='ignore')

    # Load expected feature list and scaler
    feature_list = joblib.load("models/feature_list.pkl")
    scaler = joblib.load("models/scaler.pkl")

    # Add any missing columns as zero
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0

    # Drop extra columns
    df = df[feature_list]

    # Fill any NaN with 0 (to avoid errors)
    df = df.fillna(0)

    # Scale data
    X_scaled = scaler.transform(df)

    return X_scaled
