import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "../dataset/Crop_recommendation.csv")
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
# Save model
model_path = os.path.join(BASE_DIR, "model.pkl")
joblib.dump(model, model_path)
print("Model trained and saved as model.pkl successfully!")
