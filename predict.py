import joblib
import pandas as pd

model = joblib.load("models/RandomForest.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

def predict_churn(data):

    df = pd.DataFrame([data])

    # numeric conversion
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"])
    df["tenure"] = pd.to_numeric(df["tenure"])
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

    # Encode categorical safely
    for col, encoder in label_encoders.items():

        if col != "Churn" and col in df.columns:

            df[col] = df[col].astype(str).str.lower()

            df[col] = df[col].apply(
                lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            )

            df[col] = encoder.transform(df[col])

    # remove id column
    df = df.drop("customerID", axis=1, errors="ignore")

    # ensure column order
    df = df[scaler.feature_names_in_]

    # scale
    df_scaled = scaler.transform(df)

    # prediction
    pred = model.predict(df_scaled)[0]

    # probability of churn
    prob = model.predict_proba(df_scaled)[0][1]

    probability_percent = round(prob * 100, 2)

    if pred == 1:
        result = "Customer WILL Churn"
    else:
        result = "Customer will NOT Churn"

    return result, probability_percent