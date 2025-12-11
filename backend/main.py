from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
import numpy as np
import requests

app = FastAPI(title="🌾 Smart Farming API — Crop, Weather, Irrigation, Yield, Market Insights")

# ===============================
# ✅ CORS Setup
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# 🧠 Load Crop Recommendation Model
# ===============================
model_path = "model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ model.pkl not found. Run model_train.py first.")
model = joblib.load(model_path)

# ===============================
# 🧠 Load Crop Yield Model
# ===============================
yield_model_path = "yield_model.pkl"
yield_columns_path = "yield_model_columns.pkl"

yield_model = None
yield_model_columns = None
if os.path.exists(yield_model_path) and os.path.exists(yield_columns_path):
    yield_model = joblib.load(yield_model_path)
    yield_model_columns = joblib.load(yield_columns_path)
    print("✅ Crop Yield model loaded successfully")
else:
    print("⚠️ Crop Yield model not found. Run model_train_yield.py first.")

# ===============================
# 🌦️ OpenWeather API Config
# ===============================
OPENWEATHER_API_KEY = "0adc9129d6eb63b80d6b028c9bf0e2ba"
CITY = "Coimbatore"
UNITS = "metric"

# ===============================
# 🏪 Agmarknet API Config
# ===============================
AGMARKNET_API_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
AGMARKNET_URL = (
    "https://api.data.gov.in/resource/"
    "9ef84268-d588-465a-a308-a864a43d0070"
)

# ===============================
# 📡 Root Route
# ===============================
@app.get("/")
def home():
    return {"message": "🌾 Smart Farming API is running!"}

# ===============================
# 📊 Get Latest Telemetry Data
# ===============================
@app.get("/latest")
def latest_telemetry():
    telemetry_path = "telemetry.csv"
    if not os.path.exists(telemetry_path):
        raise HTTPException(status_code=404, detail="telemetry.csv not found.")
    telemetry = pd.read_csv(telemetry_path, on_bad_lines="skip")
    if telemetry.empty:
        raise HTTPException(status_code=400, detail="Telemetry file is empty.")
    latest = telemetry.iloc[-1]

    def safe_json_value(val):
        if pd.isna(val):
            return None
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
        return val

    result = {col: safe_json_value(latest[col]) for col in telemetry.columns}
    return result

# ===============================
# 🌱 Crop Recommendation Endpoint
# ===============================
@app.get("/predict")
def predict_crop():
    telemetry_path = "telemetry.csv"
    if not os.path.exists(telemetry_path):
        raise HTTPException(status_code=404, detail="telemetry.csv not found.")
    telemetry = pd.read_csv(telemetry_path, on_bad_lines="skip")
    if telemetry.empty:
        raise HTTPException(status_code=400, detail="Telemetry file is empty.")
    latest = telemetry.iloc[-1]

    def safe_value(val):
        return float(val) if pd.notnull(val) else 0.0

    N = safe_value(latest.get("N"))
    P = safe_value(latest.get("P"))
    K = safe_value(latest.get("K"))
    temperature = safe_value(latest.get("temperature"))
    humidity = safe_value(latest.get("humidity"))
    ph = safe_value(latest.get("ph"))
    rainfall = safe_value(latest.get("rainfall"))
    soil_moisture = safe_value(latest.get("soil_moisture"))

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    try:
        prediction = model.predict(features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    telemetry_dict = {
        "timestamp": latest.get("timestamp"),
        "device_id": latest.get("device_id"),
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
        "soil_moisture": soil_moisture
    }

    return {
        "telemetry_data": telemetry_dict,
        "recommended_crop": str(prediction)
    }

# ===============================
# 🌦️ Weather Endpoint
# ===============================
@app.get("/weather")
def get_weather():
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={OPENWEATHER_API_KEY}&units={UNITS}"
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch weather data")

        weather_info = {
            "city": data.get("name"),
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "rainfall": data.get("rain", {}).get("1h", 0.0),
            "condition": data["weather"][0]["description"]
        }
        return weather_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather API error: {str(e)}")

# ===============================
# 💧 Irrigation Alert Endpoint
# ===============================
@app.get("/irrigation-alert")
def irrigation_alert():
    telemetry_path = "telemetry.csv"
    if not os.path.exists(telemetry_path):
        raise HTTPException(status_code=404, detail="telemetry.csv not found.")
    telemetry = pd.read_csv(telemetry_path, on_bad_lines="skip")
    if telemetry.empty:
        raise HTTPException(status_code=400, detail="Telemetry file is empty.")
    latest = telemetry.iloc[-1]
    soil_moisture = float(latest.get("soil_moisture", 0.0) or 0.0)

    weather = get_weather()
    rainfall = weather.get("rainfall", 0.0)

    if soil_moisture < 30 and rainfall < 2:
        alert_message = "🚨 Low soil moisture and no rain — Irrigation needed immediately!"
    elif soil_moisture < 30 and rainfall >= 2:
        alert_message = "💧 Low soil moisture but rain expected — Irrigation can be delayed."
    else:
        alert_message = "✅ Soil moisture is sufficient — No irrigation needed."

    return {
        "soil_moisture": soil_moisture,
        "rainfall": rainfall,
        "alert_message": alert_message
    }

# ===============================
# 🌾 Crop Yield Prediction Endpoint
# ===============================
@app.get("/yield-predict")
def yield_predict(
    crop: str = Query(...),
    state: str = Query(...),
    season: str = Query(...),
    area: float = Query(...),
    rainfall: float = Query(...),
    fertilizer: float = Query(...),
    pesticide: float = Query(...),
):
    if yield_model is None or yield_model_columns is None:
        raise HTTPException(status_code=500, detail="Crop Yield model not loaded. Run model_train_yield.py first.")

    input_dict = {
        "Crop": [crop],
        "State": [state],
        "Season": [season],
        "Area": [area],
        "Annual_Rainfall": [rainfall],
        "Fertilizer": [fertilizer],
        "Pesticide": [pesticide],
    }

    df = pd.DataFrame(input_dict)
    df = pd.get_dummies(df)
    df = df.reindex(columns=yield_model_columns, fill_value=0)

    try:
        predicted_yield = yield_model.predict(df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yield prediction failed: {str(e)}")

    total_production = predicted_yield * area

    return {
        "predicted_yield": round(float(predicted_yield), 2),
        "total_production": round(float(total_production), 2),
        "yield_unit": "tons per hectare",
        "production_unit": "tons"
    }

# ===============================
# 🆕 Auto-fill Endpoint for yield.html
# ===============================
@app.get("/latest-sensor-data")
def latest_sensor_data():
    telemetry_path = "telemetry.csv"
    if not os.path.exists(telemetry_path):
        raise HTTPException(status_code=404, detail="telemetry.csv not found.")
    telemetry = pd.read_csv(telemetry_path, on_bad_lines="skip")
    if telemetry.empty:
        raise HTTPException(status_code=400, detail="Telemetry file is empty.")
    latest = telemetry.iloc[-1]

    def safe_value(val):
        return float(val) if pd.notnull(val) else 0.0

    N = safe_value(latest.get("N"))
    P = safe_value(latest.get("P"))
    K = safe_value(latest.get("K"))
    temperature = safe_value(latest.get("temperature"))
    humidity = safe_value(latest.get("humidity"))
    ph = safe_value(latest.get("ph"))
    rainfall = safe_value(latest.get("rainfall"))

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    try:
        crop_prediction = model.predict(features)[0]
    except Exception:
        crop_prediction = ""

    return {
        "crop": str(crop_prediction),
        "state": "Tamil Nadu",
        "season": "Kharif",
        "rainfall": rainfall,
        "fertilizer": 40,
        "pesticide": 15
    }

# ===============================================================
# ✅ MARKET INSIGHTS — USING LOCAL DATASET ONLY (UPDATED)
# ===============================================================

# Reuse existing imports: os, pandas as pd are already imported above.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MARKET_DATA_PATH = os.path.join(BASE_DIR, "..", "dataset", "market.csv")

if not os.path.exists(MARKET_DATA_PATH):
    raise FileNotFoundError(f"❌ market.csv NOT FOUND at {MARKET_DATA_PATH}")

# Load dataset once at startup
market_df = pd.read_csv(MARKET_DATA_PATH)

# Handle Excel's encoded space columns if present
market_df = market_df.rename(columns={
    "Min_x0020_Price": "Min_Price",
    "Max_x0020_Price": "Max_Price",
    "Modal_x0020_Price": "Modal_Price",
})

# Ensure final schema (if your CSV already has these names, this is a no-op)
expected_cols = [
    "State", "District", "Market", "Commodity", "Variety",
    "Grade", "Arrival_Date", "Min_Price", "Max_Price", "Modal_Price"
]
# If any expected column is missing due to different header casing, you can add more mappings here.

# Types
for c in ["Min_Price", "Max_Price", "Modal_Price"]:
    if c in market_df.columns:
        market_df[c] = pd.to_numeric(market_df[c], errors="coerce")

# Parse dates (dataset looks like dd/mm/YYYY)
if "Arrival_Date" in market_df.columns:
    market_df["Arrival_Date"] = pd.to_datetime(
        market_df["Arrival_Date"], errors="coerce", dayfirst=True
    )

# Drop rows without modal price for consistent charts/tables
if "Modal_Price" in market_df.columns:
    market_df = market_df.dropna(subset=["Modal_Price"]).reset_index(drop=True)

@app.get("/market_meta")
def market_meta():
    """
    Lightweight endpoint to populate dropdowns quickly (states, districts, commodities, date range).
    """
    states = sorted([s for s in market_df["State"].dropna().unique().tolist()]) if "State" in market_df.columns else []
    districts = sorted([s for s in market_df["District"].dropna().unique().tolist()]) if "District" in market_df.columns else []
    commodities = sorted([s for s in market_df["Commodity"].dropna().unique().tolist()]) if "Commodity" in market_df.columns else []

    date_min = None
    date_max = None
    if "Arrival_Date" in market_df.columns and market_df["Arrival_Date"].notna().any():
        date_min = str(market_df["Arrival_Date"].min().date())
        date_max = str(market_df["Arrival_Date"].max().date())

    return {
        "status": "success",
        "total": int(len(market_df)),
        "states": states,
        "districts": districts,
        "commodities": commodities,
        "date_min": date_min,
        "date_max": date_max,
    }

@app.get("/market_insights")
def get_market_insights(
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    state: str = Query("ALL"),
    district: str = Query("ALL"),
    commodity: str = Query("ALL"),
    date_from: str = Query(None, description="YYYY-MM-DD"),
    date_to: str = Query(None, description="YYYY-MM-DD"),
):
    """
    Paginated + filterable market data.
    Query params:
      - page, page_size
      - state, district, commodity (use 'ALL' to disable filter)
      - date_from, date_to (YYYY-MM-DD)
    """
    sub = market_df

    # exact-match filters (case-insensitive), keep ALL as no-op
    if "State" in sub.columns and state and state.upper() != "ALL":
        sub = sub[sub["State"].str.lower() == state.strip().lower()]
    if "District" in sub.columns and district and district.upper() != "ALL":
        sub = sub[sub["District"].str.lower() == district.strip().lower()]
    if "Commodity" in sub.columns and commodity and commodity.upper() != "ALL":
        sub = sub[sub["Commodity"].str.lower() == commodity.strip().lower()]

    # date range
    if "Arrival_Date" in sub.columns:
        if date_from:
            try:
                sub = sub[sub["Arrival_Date"] >= pd.to_datetime(date_from)]
            except Exception:
                pass
        if date_to:
            try:
                sub = sub[sub["Arrival_Date"] <= pd.to_datetime(date_to)]
            except Exception:
                pass

    total = int(len(sub))

    # Pagination
    start = (page - 1) * page_size
    end = start + page_size
    page_df = sub.iloc[start:end].copy()

    # Make dates JSON-friendly
    if "Arrival_Date" in page_df.columns:
        page_df["Arrival_Date"] = page_df["Arrival_Date"].dt.strftime("%Y-%m-%d")

    rows = page_df.to_dict(orient="records")

    return {
        "status": "success",
        "source": "local",
        "page": page,
        "page_size": page_size,
        "total": total,
        "columns": list(page_df.columns),
        "rows": rows,
    }
