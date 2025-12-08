# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

# Load dataset (downloaded from app assets)
df = pd.read_csv("data/pagasa_synthetic_monthly.csv")  # or /path/to/pagasa_synthetic_monthly.csv

# Features
X = df[["Year", "Month", "Station"]].copy()
X["month_sin"] = np.sin(2 * np.pi * X["Month"] / 12)
X["month_cos"] = np.cos(2 * np.pi * X["Month"] / 12)
y = df["Rainfall_mm"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessor
cat_cols = ["Station"]
num_cols = ["Year", "Month", "month_sin", "month_cos"]
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])

# Pipeline
pipeline = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "models/rainfall_model.joblib")

# Quick eval
print("Train R2:", pipeline.score(X_train, y_train))
print("Test R2:", pipeline.score(X_test, y_test))
