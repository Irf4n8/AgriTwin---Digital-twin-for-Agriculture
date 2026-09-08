import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Dataset path (same folder as crop recommendation)
dataset_path = "../dataset/crop_yield.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"{dataset_path} not found. Please download the dataset.")

# Load the dataset
data = pd.read_csv(dataset_path)

# ✅ Features and target (Yield is the target)
X = data.drop(columns=["Yield"])
y = data["Yield"]

# Convert categorical columns (like Crop, State, Season) to numeric (one-hot encoding)
X = pd.get_dummies(X, drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model and column info for preprocessing during prediction
joblib.dump(model, "yield_model.pkl")
joblib.dump(X.columns.tolist(), "yield_model_columns.pkl")

print("✅ Crop Yield Prediction model trained and saved as yield_model.pkl")
