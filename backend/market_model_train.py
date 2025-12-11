# market_model_train.py — trains RandomForest on Modal_Price
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (BASE_DIR / ".." / "dataset" / "market.csv").resolve()
MODEL_DIR = (BASE_DIR / "models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Normalize Excel’s encoded spaces
df = df.rename(columns={
    "Min_x0020_Price": "Min_Price",
    "Max_x0020_Price": "Max_Price",
    "Modal_x0020_Price": "Modal_Price",
})

# Ensure final schema
df.columns = [
    "State", "District", "Market", "Commodity", "Variety",
    "Grade", "Arrival_Date", "Min_Price", "Max_Price", "Modal_Price"
]

# Numeric conversions
for c in ["Modal_Price", "Min_Price", "Max_Price"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["Modal_Price"]).reset_index(drop=True)

# Features/target
X = df[["State", "District", "Market", "Commodity", "Variety", "Min_Price", "Max_Price"]]
y = df["Modal_Price"]

# One-hot encode categoricals
X_enc = pd.get_dummies(X)
columns = X_enc.columns

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Save
joblib.dump(model, MODEL_DIR / "market_model.pkl")
joblib.dump(list(columns), MODEL_DIR / "market_columns.pkl")

print("✅ Market Price Prediction Model trained and saved to ./models/")
