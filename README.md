# Customer Churn Prediction

## Overview

This project is an end-to-end machine learning solution to predict customer churn for a telecommunications company. The goal is to identify customers who are likely to cancel their service, allowing the business to take proactive measures to retain them.

The project demonstrates a full data science workflow, from data cleaning and exploratory analysis to model training, evaluation, and providing actionable business insights.

## Project Files

* `churn_prediction.py`: The main script that handles data preprocessing, trains two machine learning models (Logistic Regression and Random Forest), evaluates their performance, and saves the trained models.
* `test_model.py`: A simple script that loads the saved models and uses them to make a churn prediction for a new, single customer.
* `churn.csv`: The dataset containing customer information, used to train the models.
* `logistic_model.pkl`: The trained Logistic Regression model, saved as a binary file.
* `random_forest_model.pkl`: The trained Random Forest model, saved as a binary file.
* `README.md`: This file, which provides an overview and documentation of the project.

## Methodology

### 1. Data Cleaning and Preprocessing
The raw dataset was cleaned and prepared for modeling. This involved:
* Converting the `TotalCharges` column to a numeric data type.
* Handling missing values by filling them with a default value.
* Encoding categorical variables (e.g., `Contract`, `PaymentMethod`) into a numerical format suitable for machine learning models.

### 2. Exploratory Data Analysis (EDA)
Several visualizations were created to understand the data, including:
* The distribution of churned vs. non-churned customers.
* The relationship between contract type and churn rate.
* The distribution of monthly charges.

### 3. Model Training and Evaluation
Two models were trained and compared:
* **Logistic Regression:** A simple, linear model used as a baseline.
* **Random Forest Classifier:** A more robust, non-linear model that typically offers higher accuracy by combining multiple decision trees.

Both models were evaluated on an unseen test dataset using metrics like **accuracy**, a **classification report**, and a **confusion matrix** to ensure reliable performance.

### 4. Key Insights and Recommendations

Based on the feature importance analysis from the Random Forest model, several key factors were identified as strong predictors of churn:

* **Tenure, Monthly Charges, and Total Charges:** These are the most significant predictors. Customers with a short tenure and low total charges are at the highest risk of leaving.
* **Fiber Optic Service:** Customers with this internet service are more likely to churn, suggesting a need to investigate potential service or support issues.
* **Electronic Check Payment:** This payment method is correlated with a higher churn rate, which may indicate a less committed customer base.

**Recommendations:**
* Offer incentives for month-to-month customers and those with fiber optic service to encourage them to switch to a more stable payment method or contract type.
* Proactively engage with new customers to ensure a positive experience and reduce early-stage churn.
* Investigate and address any underlying issues related to the fiber optic service.

## How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Install the required libraries:**
    ```bash
    pip install pandas scikit-learn seaborn matplotlib joblib
    ```
3.  **Train and save the models:**
    ```bash
    python churn_prediction.py
    ```
4.  **Make a prediction:**
    ```bash
    python test_model.py
    ```