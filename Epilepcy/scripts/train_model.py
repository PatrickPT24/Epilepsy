import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load dataset
df = pd.read_csv('Epilepcy/data/epileptic_seizure_data.csv')

# Drop unnamed columns if present
df = df.drop(columns=['Unnamed', 'Unnamed: 0'], errors='ignore')

# Convert multi-class to binary: 1 = seizure, 0 = non-seizure
df['y'] = df['y'].apply(lambda x: 1 if x == 1 else 0)

# Split features and target
X = df.drop('y', axis=1)
y = df['y']

# Save feature list
feature_list = list(X.columns)
joblib.dump(feature_list, 'Epilepcy/models/feature_list.pkl')

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'Epilepcy/models/scaler.pkl')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
rf = RandomForestClassifier(n_estimators=100, random_state=42)

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# Ensemble model
ensemble = VotingClassifier(estimators=[
    ('rf', rf),
    ('xgb', xgb)
], voting='soft')

# Train models
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
ensemble.fit(X_train, y_train)

# Save models
joblib.dump(rf, 'Epilepcy/models/rf_model.pkl')
joblib.dump(xgb, 'Epilepcy/models/xgb_model.pkl')
joblib.dump(ensemble, 'Epilepcy/models/ensemble_model.pkl')

# Evaluate accuracy
y_pred = ensemble.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("✅ Ensemble Model Accuracy:", round(accuracy * 100, 2), "%")
