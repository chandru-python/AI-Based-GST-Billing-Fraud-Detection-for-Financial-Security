import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# ---- 1. Load dataset ----
df = pd.read_csv("dataset.csv")

# ---- 2. Define sample size ----
n = 100000

# ---- 3. Sample target=1 and target=0 ----
count_1 = len(df[df['target'] == 1])
df_1 = df[df['target'] == 1].sample(n=n, replace=(count_1 < n), random_state=42)

count_0 = len(df[df['target'] == 0])
df_0 = df[df['target'] == 0].sample(n=n, replace=(count_0 < n), random_state=42)

# ---- 4. Combine sampled data ----
df_balanced = pd.concat([df_1, df_0]).reset_index(drop=True)

# ---- 5. Separate features and target ----
X = df_balanced.drop('target', axis=1)
y = df_balanced['target']

# ---- 6. Identify numeric and categorical columns ----
num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# ---- 7. Impute missing values ----
# Numeric: mean
imputer_num = SimpleImputer(strategy='mean')
X[num_cols] = imputer_num.fit_transform(X[num_cols])

# Categorical: mode
imputer_cat = SimpleImputer(strategy='most_frequent')
X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])

# ---- 8. Encode categorical columns ----
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # store for inverse transform if needed

# ---- 9. Check final dataset ----
print("Features shape:", X.shape)
print("Target distribution:\n", y.value_counts())
print("Any remaining NAs in X:\n", X.isna().sum())

# ---- 10. Save processed data ----
df_final = pd.concat([X, y], axis=1)
df_final.to_csv("processed_balanced_dataset.csv", index=False)
print("Processed dataset saved as 'processed_balanced_dataset.csv'")
