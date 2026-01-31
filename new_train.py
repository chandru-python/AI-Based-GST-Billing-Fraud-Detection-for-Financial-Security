# ======================================================
# IMPORT LIBRARIES
# ======================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, f1_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import matplotlib.pyplot as plt
import joblib

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv("balanced.csv")

# Drop ID columns automatically
id_cols = [col for col in df.columns if "ID" in col or "TransactionID" in col]
X = df.drop(columns=id_cols + ["Fraudulent_Label"], errors='ignore')
y = df["Fraudulent_Label"]

# ======================================================
# TRAIN-TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================================================
# DATA PREPROCESSING
# ======================================================
# Identify numeric and categorical columns
num_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Impute numeric columns
if num_cols:
    imputer_num = SimpleImputer(strategy='mean')
    X_train.loc[:, num_cols] = imputer_num.fit_transform(X_train[num_cols])
    X_test.loc[:, num_cols] = imputer_num.transform(X_test[num_cols])

# Impute and encode categorical columns
label_encoders = {}
if cat_cols:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X_train.loc[:, cat_cols] = imputer_cat.fit_transform(X_train[cat_cols])
    X_test.loc[:, cat_cols] = imputer_cat.transform(X_test[cat_cols])

    for col in cat_cols:
        le = LabelEncoder()
        X_train.loc[:, col] = le.fit_transform(X_train[col])
        X_test.loc[:, col] = le.transform(X_test[col])
        label_encoders[col] = le

# ======================================================
# FUNCTION TO TRAIN, EVALUATE, AND RETURN MODEL
# ======================================================
def train_evaluate_xgb(X_tr, y_tr, X_te, y_te, strategy_name):
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_pred_proba = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_pred_proba)
    recall_fraud = recall_score(y_te, y_pred, pos_label=1)
    f1_fraud = f1_score(y_te, y_pred, pos_label=1)

    print(f"\n========== {strategy_name} ==========")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Recall (Fraud=1): {recall_fraud:.4f}")
    print(f"F1-Score (Fraud=1): {f1_fraud:.4f}")
    print(classification_report(y_te, y_pred))

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_te, y_pred_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0,1],[0,1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {strategy_name}")
    plt.legend(loc="lower right")
    plt.show()

    return model, acc, recall_fraud, f1_fraud

# ======================================================
# RUN ALL STRATEGIES
# ======================================================
# 1. Base XGBoost
xgb_base, acc_base, recall_base, f1_base = train_evaluate_xgb(X_train, y_train, X_test, y_test, "Base XGBoost")

# 2. Undersampling
undersampler = RandomUnderSampler(random_state=42)
X_under, y_under = undersampler.fit_resample(X_train, y_train)
xgb_under, acc_under, recall_under, f1_under = train_evaluate_xgb(X_under, y_under, X_test, y_test, "Undersampling XGBoost")

# 3. Oversampling
oversampler = RandomOverSampler(random_state=42)
X_over, y_over = oversampler.fit_resample(X_train, y_train)
xgb_over, acc_over, recall_over, f1_over = train_evaluate_xgb(X_over, y_over, X_test, y_test, "Oversampling XGBoost")

# 4. SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
xgb_smote, acc_smote, recall_smote, f1_smote = train_evaluate_xgb(X_smote, y_smote, X_test, y_test, "SMOTE XGBoost")

# ======================================================
# SELECT AND SAVE BEST MODEL (BASED ON F1 FRAUD)
# ======================================================
models_dict = {
    "base": (xgb_base, acc_base, recall_base, f1_base),
    "undersampling": (xgb_under, acc_under, recall_under, f1_under),
    "oversampling": (xgb_over, acc_over, recall_over, f1_over),
    "smote": (xgb_smote, acc_smote, recall_smote, f1_smote),
}

best_model_name = max(models_dict, key=lambda k: models_dict[k][3])  # based on F1 fraud
best_model, best_acc, best_recall, best_f1 = models_dict[best_model_name]

print(f"\n✅ Best Model: {best_model_name}")
print(f"   Accuracy: {best_acc:.4f}")
print(f"   Recall (Fraud=1): {best_recall:.4f}")
print(f"   F1-Score (Fraud=1): {best_f1:.4f}")

# Save the best model
joblib.dump(best_model, f"best_xgb_model_{best_model_name}.pkl")
print(f"Model saved as: best_xgb_model_{best_model_name}.pkl")
