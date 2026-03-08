import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Load dataset
df = pd.read_csv("churn_data.csv")

# Convert TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.dropna(inplace=True)

categorical_cols = [
"gender","Partner","Dependents","PhoneService","MultipleLines",
"InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
"TechSupport","StreamingTV","StreamingMovies","Contract",
"PaperlessBilling","PaymentMethod","Churn"
]

label_encoders = {}

# Encode categorical
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str).str.lower())
    label_encoders[col] = le

# Features & Target
X = df.drop(["customerID","Churn"],axis=1)
y = df["Churn"]

# Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier()
model.fit(X_train,y_train)

print(classification_report(y_test,model.predict(X_test)))

# Save files
os.makedirs("models",exist_ok=True)

joblib.dump(model,"models/RandomForest.pkl")
joblib.dump(scaler,"models/scaler.pkl")
joblib.dump(label_encoders,"models/label_encoders.pkl")

print("Model Saved Successfully")