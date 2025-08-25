# Step 1: Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
# Step 2: Load dataset
df = pd.read_csv("churn.csv")

# Keep a copy of the original dataset for visualization (EDA)
raw_df = df.copy()

# Step 3: Explore dataset
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nInfo:\n")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# Step 4: Data Cleaning & Preprocessing
# Convert 'TotalCharges' to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Encode target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop customerID
df = df.drop("customerID", axis=1)

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)

print("\n✅ Cleaned dataset shape:", df.shape)
print(df.head())

# Step 5: Split Data
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n✅ Logistic Regression Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)
# Step 7: Random Forest Model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n✅ Random Forest Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

import joblib

# Save models
joblib.dump(model, "logistic_model.pkl")
joblib.dump(rf_model, "random_forest_model.pkl")
print("\n✅ Models saved as logistic_model.pkl and random_forest_model.pkl")

# Example: Load and test Logistic Regression model
loaded_model = joblib.load("logistic_model.pkl")
sample_pred = loaded_model.predict(X_test[:5])
print("\n🔮 Predictions from loaded Logistic Regression model:", sample_pred.tolist())
print("Actual:", y_test[:5].tolist())


# Step 8: Exploratory Data Analysis (using raw_df)
# Churn distribution
plt.figure(figsize=(5,4))
sns.countplot(x="Churn", data=raw_df)
plt.title("Churn Distribution")
plt.show()

# Churn vs Contract Type
plt.figure(figsize=(6,4))
sns.countplot(x="Contract", hue="Churn", data=raw_df)
plt.title("Churn vs Contract Type")
plt.show()

# Monthly Charges distribution
plt.figure(figsize=(6,4))
sns.histplot(data=raw_df, x="MonthlyCharges", bins=30, kde=True, hue="Churn")
plt.title("Monthly Charges vs Churn")
plt.show()

# Step 9: Feature Importance (from Logistic Regression)
importance = pd.Series(model.coef_[0], index=X.columns)
top_features = importance.abs().sort_values(ascending=False).head(10)

plt.figure(figsize=(8,6))
top_features.plot(kind="barh", color="teal")
plt.title("Top 10 Features Influencing Churn (Logistic Regression)")
plt.show()

