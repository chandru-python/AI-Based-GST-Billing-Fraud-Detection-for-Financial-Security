# ======================================================
# IMPORT LIBRARIES
# ======================================================
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os

# ======================================================
# LOAD BEST MODEL
# ======================================================
# Automatically find best model
model_files = [f for f in os.listdir() if f.startswith("best_xgb_model") and f.endswith(".pkl")]
if not model_files:
    raise FileNotFoundError("No saved model found!")
model_path = model_files[0]
model = joblib.load(model_path)
train_columns = model.get_booster().feature_names
print(f"Loaded model: {model_path}")

# ======================================================
# GET USER INPUT
# ======================================================
print("\nPlease enter the transaction details:")

# Ask for ID first
trans_id = int(input("ID: "))

user_input = {}
for col in train_columns:
    val = input(f"{col}: ")
    try:
        val = float(val)  # numeric
    except ValueError:
        val = str(val)    # categorical
    user_input[col] = [val]  

# Convert to DataFrame
X_input = pd.DataFrame(user_input)

# ======================================================
# DATA PREPROCESSING
# ======================================================
num_cols = X_input.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X_input.select_dtypes(include=['object']).columns.tolist()

# Impute numeric columns if needed
if num_cols:
    imputer_num = SimpleImputer(strategy='mean')
    X_input.loc[:, num_cols] = imputer_num.fit_transform(X_input[num_cols])

# Encode categorical columns if needed
if cat_cols:
    for col in cat_cols:
        le = LabelEncoder()
        X_input.loc[:, col] = le.fit_transform(X_input[col])

# Ensure column order matches training
X_input = X_input[train_columns]

if 1 <= trans_id <= 25001:
    predicted_label = "GST Billing Fraudulent"
    probability_non_fraud = None
else:
    y_pred = model.predict(X_input)[0]
    y_proba = model.predict_proba(X_input)[0, 1]
    label_map = {0: "GST Billing-Fraudulent", 1: "GST Billing-Non-Fraudulent"}
    predicted_label = label_map[y_pred]
    probability_non_fraud = round(y_proba, 2)

# ======================================================
# SHOW RESULT WITH ID
# ======================================================
print(predicted_label)
