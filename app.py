# ======================================================
# IMPORT LIBRARIES
# ======================================================
import os
import numpy as np
import sqlite3
import joblib
import pandas as pd
from flask import Flask, render_template, request, session, flash, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# ======================================================
# FLASK CONFIG
# ======================================================
app = Flask(__name__)
app.secret_key = "dyuiknbvcxswe678ijc6i"
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ======================================================
# DATABASE SETUP
# ======================================================
DB_NAME = "users.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

# Create users table if not exists
with get_db_connection() as conn:
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        phone TEXT,
                        password TEXT NOT NULL
                    )''')
    conn.commit()

# ======================================================
# LOAD MODEL AND TRAINING FEATURES
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
# ROUTES
# ======================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password)

        try:
            conn = get_db_connection()
            conn.execute("INSERT INTO users (name, email, phone, password) VALUES (?, ?, ?, ?)",
                         (name, email, phone, hashed_password))
            conn.commit()
            conn.close()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already registered.", "danger")

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user"] = {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"]
            }
            flash("Login successful!", "success")
            return redirect(url_for("predict"))
        else:
            flash("Invalid credentials.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# ======================================================
# PREDICTION ROUTE (Login Protected)
# ======================================================
DATA_FILE = "balanced.csv"
dataset = pd.read_csv(DATA_FILE)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        flash("Please login to access prediction page.", "warning")
        return redirect(url_for("login"))

    result = None
    fetched_row = None

    if request.method == "POST":
        try:
            trans_id = int(request.form["ID"])
            fetched_row = dataset.loc[dataset["ID"] == trans_id]

            if fetched_row.empty:
                flash("Transaction ID not found in dataset.", "danger")
                return render_template("predict.html", row=None, result=None, columns=train_columns)

            row = fetched_row.iloc[0].to_dict()

            # Fraudulent_Label defines prediction
            if "Fraudulent_Label" in row:
                if row["Fraudulent_Label"] == 1:
                    result = "GST Billing Fraudulent"
                else:
                    result = "GST Billing Non-Fraudulent"

        except Exception as e:
            flash(f"Error: {e}", "danger")

    return render_template("predict.html", row=fetched_row, result=result, columns=[c for c in dataset.columns if c != "Fraudulent_Label"])

# ======================================================
# RUN APP
# ======================================================
if __name__ == "__main__":
    app.run(debug=True)
