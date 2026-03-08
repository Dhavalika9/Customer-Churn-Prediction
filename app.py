from flask import Flask, render_template, request
from predict import predict_churn

app = Flask(__name__)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "customerID": request.form.get("customerID"),
        "gender": request.form.get("gender"),
        "SeniorCitizen": request.form.get("SeniorCitizen"),
        "Partner": request.form.get("Partner"),
        "Dependents": request.form.get("Dependents"),
        "tenure": request.form.get("tenure"),
        "PhoneService": request.form.get("PhoneService"),
        "MultipleLines": request.form.get("MultipleLines"),
        "InternetService": request.form.get("InternetService"),
        "OnlineSecurity": request.form.get("OnlineSecurity"),
        "OnlineBackup": request.form.get("OnlineBackup"),
        "DeviceProtection": request.form.get("DeviceProtection"),
        "TechSupport": request.form.get("TechSupport"),
        "StreamingTV": request.form.get("StreamingTV"),
        "StreamingMovies": request.form.get("StreamingMovies"),
        "Contract": request.form.get("Contract"),
        "PaperlessBilling": request.form.get("PaperlessBilling"),
        "PaymentMethod": request.form.get("PaymentMethod"),
        "MonthlyCharges": request.form.get("MonthlyCharges"),
        "TotalCharges": request.form.get("TotalCharges")
    }

    result = predict_churn(data)

    # unpack tuple
    prediction = result[0]
    probability = result[1]

    return render_template(
        "result.html",
        prediction=prediction,
        probability=round(probability,2)
    )
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
