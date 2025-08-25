import pandas as pd
import joblib

# Load saved models
logistic_model = joblib.load("logistic_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

# Example customer details (must match training features!)
# A comprehensive dictionary of all features with default values of 0
example = {
    "SeniorCitizen": 0,
    "tenure": 0,
    "MonthlyCharges": 0.0,
    "TotalCharges": 0.0,
    "gender_Male": 0,
    "Partner_Yes": 0,
    "Dependents_Yes": 0,
    "PhoneService_Yes": 0,
    "MultipleLines_No phone service": 0,
    "MultipleLines_Yes": 0,
    "InternetService_Fiber optic": 0,
    "InternetService_No": 0,
    "OnlineSecurity_No internet service": 0,
    "OnlineSecurity_Yes": 0,
    "OnlineBackup_No internet service": 0,
    "OnlineBackup_Yes": 0,
    "DeviceProtection_No internet service": 0,
    "DeviceProtection_Yes": 0,
    "TechSupport_No internet service": 0,
    "TechSupport_Yes": 0,
    "StreamingTV_No internet service": 0,
    "StreamingTV_Yes": 0,
    "StreamingMovies_No internet service": 0,
    "StreamingMovies_Yes": 0,
    "Contract_One year": 0,
    "Contract_Two year": 0,
    "PaperlessBilling_Yes": 0,
    "PaymentMethod_Credit card (automatic)": 0,
    "PaymentMethod_Electronic check": 0,
    "PaymentMethod_Mailed check": 0
}

# Update the values for the specific example customer
example["tenure"] = 12
example["MonthlyCharges"] = 70.0
example["TotalCharges"] = 840.0
example["gender_Male"] = 1
example["PhoneService_Yes"] = 1
example["PaperlessBilling_Yes"] = 1
example["InternetService_Fiber optic"] = 1
example["PaymentMethod_Electronic check"] = 1

# Convert to DataFrame
input_data = pd.DataFrame([example])

# Predictions
logistic_pred = logistic_model.predict(input_data)[0]
rf_pred = rf_model.predict(input_data)[0]

print("✅ Logistic Regression Prediction:", "Churn" if logistic_pred == 1 else "No Churn")
print("✅ Random Forest Prediction:", "Churn" if rf_pred == 1 else "No Churn")
