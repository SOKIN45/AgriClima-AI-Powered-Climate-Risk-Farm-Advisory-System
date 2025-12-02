import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib


class ClimateModel:
    def __init__(self):
        self.rainfall_model = None
        self.temperature_model = None
        self.scaler = StandardScaler()
        self.feature_columns = None

    def prepare_features(self, X):
        """Scale features"""
        return self.scaler.fit_transform(X)

    def train_models(self, X, y_rainfall, y_temperature):
        """Train both rainfall and temperature models"""
        # Scale features
        X_scaled = self.prepare_features(X)
        self.feature_columns = X.columns.tolist()

        # Split data
        X_train, X_test, y_rain_train, y_rain_test = train_test_split(
            X_scaled, y_rainfall, test_size=0.2, random_state=42
        )

        _, _, y_temp_train, y_temp_test = train_test_split(
            X_scaled, y_temperature, test_size=0.2, random_state=42
        )

        # Train Rainfall Models
        print("Training Rainfall Models...")
        rf_rain = RandomForestRegressor(n_estimators=100, random_state=42)
        lr_rain = LinearRegression()

        rf_rain.fit(X_train, y_rain_train)
        lr_rain.fit(X_train, y_rain_train)

        # Evaluate Rainfall Models
        rf_rain_pred = rf_rain.predict(X_test)
        lr_rain_pred = lr_rain.predict(X_test)

        print("Rainfall - Random Forest Performance:")
        self.evaluate_model(y_rain_test, rf_rain_pred)
        print("Rainfall - Linear Regression Performance:")
        self.evaluate_model(y_rain_test, lr_rain_pred)

        # Train Temperature Models
        print("\nTraining Temperature Models...")
        rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
        lr_temp = LinearRegression()

        rf_temp.fit(X_train, y_temp_train)
        lr_temp.fit(X_train, y_temp_train)

        # Evaluate Temperature Models
        rf_temp_pred = rf_temp.predict(X_test)
        lr_temp_pred = lr_temp.predict(X_test)

        print("Temperature - Random Forest Performance:")
        self.evaluate_model(y_temp_test, rf_temp_pred)
        print("Temperature - Linear Regression Performance:")
        self.evaluate_model(y_temp_test, lr_temp_pred)

        # Select best models based on R² score
        rf_rain_r2 = r2_score(y_rain_test, rf_rain_pred)
        lr_rain_r2 = r2_score(y_rain_test, lr_rain_pred)

        rf_temp_r2 = r2_score(y_temp_test, rf_temp_pred)
        lr_temp_r2 = r2_score(y_temp_test, lr_temp_pred)

        self.rainfall_model = rf_rain if rf_rain_r2 > lr_rain_r2 else lr_rain
        self.temperature_model = rf_temp if rf_temp_r2 > lr_temp_r2 else lr_temp

        print(f"\nSelected Rainfall Model: {'Random Forest' if rf_rain_r2 > lr_rain_r2 else 'Linear Regression'}")
        print(f"Selected Temperature Model: {'Random Forest' if rf_temp_r2 > lr_temp_r2 else 'Linear Regression'}")

        return self.rainfall_model, self.temperature_model

    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.2f}")

        return mae, rmse, r2

    def predict(self, X):
        """Make predictions for new data"""
        X_scaled = self.scaler.transform(X)
        rainfall_pred = self.rainfall_model.predict(X_scaled)
        temperature_pred = self.temperature_model.predict(X_scaled)

        return rainfall_pred, temperature_pred

    def save_models(self, filepath):
        """Save trained models"""
        model_data = {
            'rainfall_model': self.rainfall_model,
            'temperature_model': self.temperature_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)

    def load_models(self, filepath):
        """Load trained models"""
        model_data = joblib.load(filepath)
        self.rainfall_model = model_data['rainfall_model']
        self.temperature_model = model_data['temperature_model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']