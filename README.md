# Telco Customer Churn Prediction Pipeline

This repository features an end-to-end Machine Learning solution to predict customer churn using the **IBM Telco Churn Dataset**. It automates the entire workflow from raw data cleaning to a serialized, deployment-ready pipeline.

## 📌 Project Overview
The objective is to identify customers at risk of leaving the service. By utilizing high-performance classification models, the system helps businesses implement data-driven retention strategies.

### Key Features:
* **Automated Data Cleaning:** Handles missing values and corrects data types (e.g., `TotalCharges` conversion).
* **Robust Preprocessing:** Uses `ColumnTransformer` for integrated One-Hot Encoding and Standard Scaling.
* **Pipeline Architecture:** Implements Scikit-Learn `Pipeline` to ensure clean, reproducible code and prevent data leakage.
* **Hyperparameter Optimization:** Uses `GridSearchCV` to tune the Random Forest model for maximum accuracy.
* **Model Serialization:** Exports the final trained model as a `.pkl` file for easy integration.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `joblib`
