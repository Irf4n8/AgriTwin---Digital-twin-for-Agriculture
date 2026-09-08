import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../dataset/crop_yield.csv")
MODEL_PATH = os.path.join(BASE_DIR, "yield_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "yield_model_columns.pkl")

def train_model():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Feature Selection
    # Drop columns that are not predictive features or are targets (Production is derived from Yield*Area usually, or vice versa)
    # Target: Yield
    # Features: Crop, Season, State, Area, Annual_Rainfall, Fertilizer, Pesticide
    
    # Cleaning
    df = df.dropna()
    
    target = "Yield"
    features = ["Crop", "Season", "State", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
    
    X = df[features]
    y = df[target]
    
    # One-Hot Feature Encoding
    print("Encoding features...")
    X_encoded = pd.get_dummies(X, columns=["Crop", "Season", "State"], drop_first=True)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Train
    print("Training Random Forest Regressor...")
    # n_estimators=100 is standard
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Trained. MSE: {mse:.4f}, R2 Score: {r2:.4f}")
    
    # Save
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(X_encoded.columns.tolist(), COLUMNS_PATH)
    print("Done.")

if __name__ == "__main__":
    train_model()
