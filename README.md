# End-to-End-ML-Pipeline-with-Scikit-learn-Pipeline-API

Objective
Build a complete, production-ready machine learning pipeline to predict whether a telecom customer will churn (leave the service), using the IBM Telco Customer Churn dataset.

Methodology / Approach
Loaded the Telco Customer Churn dataset (7,043 rows × 21 columns) directly from a public URL.
Handled missing values in TotalCharges by converting to numeric and filling with the median.
Built a scikit-learn Pipeline with a ColumnTransformer that applies StandardScaler to numerical columns and OneHotEncoder to categorical columns.
Trained two models: Logistic Regression (baseline) and Random Forest (tuned via GridSearchCV with 3-fold cross-validation).
Exported the best model as a .pkl file using joblib and verified it by reloading and running sample predictions.
Key Results / Observations
Model	Accuracy
Logistic Regression	82.11%
Random Forest (best)	81.05%
Best Random Forest params: n_estimators=100, max_depth=10, min_samples_split=5
The model performs well on the majority class (non-churners), with precision of 0.84 and recall of 0.91.
Churner detection (minority class) is harder — F1-score of 0.59 — highlighting the class imbalance challenge.
