import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
dataset_path = "../dataset/Crop_recommendation.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"{dataset_path} not found. Please download the dataset.")

data = pd.read_csv(dataset_path)

# Split features and label
X = data.drop("label", axis=1)
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl successfully!")
