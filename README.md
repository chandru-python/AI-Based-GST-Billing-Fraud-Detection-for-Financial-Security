# 🚨 GST Billing Fraud Detection System

An intelligent **Machine Learning + Flask Web Application** designed to detect fraudulent GST billing transactions using advanced classification techniques such as **XGBoost** with multiple sampling strategies.

This system allows users to register, log in securely, and check whether a GST transaction is fraudulent by entering a Transaction ID.

---

# ✅ Project Overview

GST fraud has become a major financial threat, causing revenue loss and compliance issues. This project leverages machine learning to automatically identify suspicious transactions and assist businesses or authorities in fraud detection.

The system:

✔ Trains multiple XGBoost models
✔ Handles imbalanced datasets using SMOTE, oversampling, and undersampling
✔ Automatically selects the best-performing model
✔ Stores the trained model for deployment
✔ Provides a secure web interface for predictions

---

# 🔥 Key Features

* Machine Learning-based fraud detection
* Automatic best model selection (based on F1-score)
* User authentication system
* SQLite database integration
* Flask-powered web application
* ROC curve visualization
* Secure password hashing
* Clean UI with HTML templates

---

# 🧠 Machine Learning Workflow

### 1️⃣ Data Preprocessing

* Missing value imputation
* Label encoding for categorical features
* Automatic ID column removal

### 2️⃣ Handling Class Imbalance

The following strategies are tested:

* Base XGBoost
* Random Undersampling
* Random Oversampling
* SMOTE

The best model is selected based on **Fraud F1-score**.

---

# 🏗️ Tech Stack

### 🔹 Backend

* Python
* Flask
* SQLite

### 🔹 Machine Learning

* Scikit-learn
* XGBoost
* Imbalanced-learn

### 🔹 Frontend

* HTML
* CSS
* JavaScript

---

# 📂 Project Structure

```
gst_billing_detection_project/
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── predict.html
│   ├── about.html
│   └── contact.html
│
├── static/
│   └── uploads/
│
├── balanced.csv
├── users.db
├── best_xgb_model_*.pkl
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation Guide

## ✅ Step 1 — Clone Repository

```bash
git clone https://github.com/your-username/gst-billing-fraud-detection.git
cd gst-billing-fraud-detection
```

---

## ✅ Step 2 — Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

### Windows:

```bash
venv\Scripts\activate
```

### Mac/Linux:

```bash
source venv/bin/activate
```

---

## ✅ Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ✅ Step 4 — Train the Model

```bash
python train_model.py
```

This will generate:

```
best_xgb_model_smote.pkl
```

(or another best model depending on performance)

---

## ✅ Step 5 — Run the Flask App

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:5000/
```

---

# 🔐 Default System Flow

1️⃣ Register a new account
2️⃣ Login securely
3️⃣ Enter Transaction ID
4️⃣ Get fraud prediction instantly

---

# 📊 Model Evaluation Metrics

The model is evaluated using:

* Accuracy
* ROC-AUC Score
* Recall (Fraud Detection Power)
* F1 Score

👉 **F1-score is prioritized** to balance precision and recall for fraud detection.

---

# 🚀 Future Improvements

* Deploy on AWS / Azure
* Add real-time fraud detection
* Implement deep learning models
* Enable file upload for bulk predictions
* Improve UI with dashboards

---

# 👨‍💻 Author

**Chandru**

Machine Learning Developer | AI Enthusiast

---

# ⭐ If You Like This Project

Give it a ⭐ on GitHub — it helps others discover the project!

---

# 📜 License

This project is for educational and research purposes.
