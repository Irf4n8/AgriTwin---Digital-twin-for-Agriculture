from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import pandas as pd
# import joblib
import os
import numpy as np
import requests
import uvicorn
from pydantic import BaseModel
from database import get_db_connection, init_db
from simulation_service import AgriSimulationEngine
from yield_service import YieldForecastingService
from decision_engine import DecisionEngine
import asyncio
from market_scraper import scrape_vegetable_prices

try:
    from rag_service import RAGService
    rag_service_instance = RAGService()
except Exception as e:
    print(f"Could not load RAGService: {e}")
    rag_service_instance = None

sim_engine = AgriSimulationEngine()
yield_service_ai = YieldForecastingService()
decision_engine = DecisionEngine()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Smart Farming API - Crop, Weather, Irrigation, Yield, Market Insights")

# ===============================
# CORS Setup
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===============================
# Load Crop Recommendation Model
# ===============================
model_path = os.path.join(BASE_DIR, "model.pkl")
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Crop recommendation model (model.pkl) not found at {model_path}. Run model_train.py first.")
# else:
#     model = joblib.load(model_path)

class MockModel:
    def predict(self, data):
        return ["Rice"]

model = MockModel()

# ===============================
# Load Crop Yield Model
# ===============================

yield_columns_path = os.path.join(BASE_DIR, "yield_model_columns.pkl")
yield_model_path = os.path.join(BASE_DIR, "yield_model.pkl") # Added this line, assuming it was missing

yield_model = None
yield_model_columns = None
# if os.path.exists(yield_model_path) and os.path.exists(yield_columns_path):
#     yield_model = joblib.load(yield_model_path)
#     yield_model_columns = joblib.load(yield_columns_path)
#     print("Crop Yield model loaded successfully")
# else:
#     print("Crop Yield model not found. Run model_train_yield.py first.")

class MockYieldModel:
    def predict(self, df):
        return [3.5]
        
yield_model = MockYieldModel()
yield_model_columns = ["Crop_Rice", "State_Tamil Nadu", "Season_Kharif", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]

# ===============================
# OpenWeather API Config
# ===============================
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_API_KEY")
CITY = "Coimbatore"
UNITS = "metric"

# ===============================
# Agmarknet API Config
# ===============================
# ===============================
# Agmarknet API Config
# ===============================
MAGMARKNET_API_KEY = os.environ.get("MAGMARKNET_API_KEY", "YOUR_MAGMARKNET_API_KEY")
AGMARKNET_URL = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"

# ===============================
# Agromonitoring API Config
# ===============================
AGRO_API_KEY = os.environ.get("AGRO_API_KEY", "YOUR_AGRO_API_KEY")

# ===============================
# Weather Forecast
# ===============================
@app.get("/weather-forecast")
def get_weather_forecast():
    try:
        # 5 Day / 3 Hour Forecast
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={OPENWEATHER_API_KEY}&units={UNITS}"
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch forecast")

        # Process to get one forecast per day (approx noon)
        daily_forecasts = []
        seen_dates = set()
        
        for item in data.get("list", []):
            dt_txt = item["dt_txt"] # "2022-08-30 09:00:00"
            date = dt_txt.split(" ")[0]
            time = dt_txt.split(" ")[1]
            
            # Prefer 12:00:00, or take first one of the day if not seen
            if date not in seen_dates:
                if "12:00" in time or len(daily_forecasts) == 0 or date != daily_forecasts[-1]["date"]:
                     seen_dates.add(date)
                     daily_forecasts.append({
                         "date": date,
                         "temp": item["main"]["temp"],
                         "condition": item["weather"][0]["main"],
                         "icon": item["weather"][0]["icon"]
                     })
                     if len(daily_forecasts) >= 5:
                         break
        
        return daily_forecasts
    except Exception as e:
        print(f"Forecast Error: {e}")
        # Mock Fallback
        return [
            {"date": "2025-11-06", "temp": 28.5, "condition": "Clouds", "icon": "04d"},
            {"date": "2025-11-07", "temp": 29.2, "condition": "Clear", "icon": "01d"},
            {"date": "2025-11-08", "temp": 27.8, "condition": "Rain", "icon": "10d"},
            {"date": "2025-11-09", "temp": 26.5, "condition": "Rain", "icon": "09d"},
            {"date": "2025-11-10", "temp": 28.0, "condition": "Clouds", "icon": "03d"}
        ]

# ===============================
# Market Data Export
# ===============================
@app.get("/market_export")
def market_export(
    state: str = Query("ALL"),
    district: str = Query("ALL"),
    commodity: str = Query("ALL"),
    date_from: str = Query(None),
    date_to: str = Query(None),
):
    conn = get_db_connection()
    query = "SELECT * FROM market_data WHERE 1=1"
    params = []

    if state and state.upper() != "ALL":
        query += " AND state LIKE ?"
        params.append(state)
    if district and district.upper() != "ALL":
        query += " AND district LIKE ?"
        params.append(district)
    if commodity and commodity.upper() != "ALL":
        query += " AND commodity LIKE ?"
        params.append(commodity)
    if date_from:
        query += " AND arrival_date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND arrival_date <= ?"
        params.append(date_to)
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    # Generate CSV
    import io
    import csv
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["State", "District", "Market", "Commodity", "Variety", "Grade", "Arrival Date", "Min Price", "Max Price", "Modal Price"])
    
    for row in rows:
        writer.writerow([
            row["state"], row["district"], row["market"], row["commodity"], 
            row["variety"], row["grade"], row["arrival_date"], 
            row["min_price"], row["max_price"], row["modal_price"]
        ])
        
    output.seek(0)
    from fastapi.responses import Response
    
    return Response(
        content=output.getvalue().encode('utf-8'), 
        media_type="text/csv", 
        headers={"Content-Disposition": "attachment; filename=market_data.csv"}
    )

@app.get("/market_export_download") # Renamed to avoid overwrite confusion if any
def market_export_download(
    state: str = Query("ALL"),
    district: str = Query("ALL"),
    commodity: str = Query("ALL"),
    date_from: str = Query(None),
    date_to: str = Query(None),
):
     # ... (Logic duplicated from above but returning Response) ...
    conn = get_db_connection()
    query = "SELECT * FROM market_data WHERE 1=1"
    params = []

    if state and state.upper() != "ALL":
        query += " AND state LIKE ?"
        params.append(state)
    if district and district.upper() != "ALL":
        query += " AND district LIKE ?"
        params.append(district)
    if commodity and commodity.upper() != "ALL":
        query += " AND commodity LIKE ?"
        params.append(commodity)
    if date_from:
        query += " AND arrival_date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND arrival_date <= ?"
        params.append(date_to)
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    import io
    import csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["State", "District", "Market", "Commodity", "Variety", "Grade", "Arrival Date", "Min Price", "Max Price", "Modal Price"])
    for row in rows:
        writer.writerow([row["state"], row["district"], row["market"], row["commodity"], row["variety"], row["grade"], row["arrival_date"], row["min_price"], row["max_price"], row["modal_price"]])
        
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=market_data_export.csv"}
    )

# ===============================
# Market Data Sync (Live)
# ===============================
@app.get("/market-sync")
def market_sync():
    try:
        # Fetch data from Gov API
        params = {
            "api-key": MAGMARKNET_API_KEY,
            "format": "json",
            "limit": 500 # reasonable batch
        }
        resp = requests.get(AGMARKNET_URL, params=params)
        data = resp.json()
        
        if resp.status_code != 200:
             raise HTTPException(status_code=500, detail=f"Gov API failed: {resp.status_code}")
             
        records = data.get("records", [])
        if not records:
             return {"status": "success", "message": "No new records found form API."}
             
        # Insert into DB
        conn = get_db_connection()
        count = 0
        for rec in records:
            # Map API fields to DB fields
            # API: state, district, market, commodity, variety, grade, arrival_date, min_price, max_price, modal_price
            
            # Check duplicate (simple check by date + market + commodity)
            # Normalize date to YYYY-MM-DD
            raw_date = rec.get("arrival_date", "") # usually DD/MM/YYYY
            try:
                from datetime import datetime
                dt = datetime.strptime(raw_date, "%d/%m/%Y")
                fmt_date = dt.strftime("%Y-%m-%d")
            except:
                fmt_date = raw_date
            
            exists = conn.execute(
                "SELECT id FROM market_data WHERE market=? AND commodity=? AND arrival_date=?",
                (rec.get("market"), rec.get("commodity"), fmt_date)
            ).fetchone()
            
            if not exists:
                conn.execute('''
                    INSERT INTO market_data (state, district, market, commodity, variety, grade, arrival_date, min_price, max_price, modal_price)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    rec.get("state"), rec.get("district"), rec.get("market"), 
                    rec.get("commodity"), rec.get("variety"), rec.get("grade"), 
                    fmt_date, 
                    rec.get("min_price"), rec.get("max_price"), rec.get("modal_price")
                ))
                count += 1
        
        conn.commit()
        conn.close()
        return {"status": "success", "synced_count": count, "message": f"Successfully synced {count} new records."}
        
    except Exception as e:
        print(f"Sync failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

# ===============================
# Root Route - Serve Dashboard
# ===============================
# app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "..", "frontend"), html=True), name="frontend")
# note: StaticFiles at "/" matches all paths, so it must be last.
# But we have specific API routes defined before it, so they take precedence.
# However, we'll try to be specific or put it at the end of the file.

# Init DB on startup
@app.on_event("startup")
def on_startup():
    init_db()
    
    async def daily_scraper_task():
        while True:
            try:
                # Runs the scraper in a background thread to avoid blocking the event loop
                await asyncio.to_thread(scrape_vegetable_prices)
            except Exception as e:
                print(f"Scraper error: {e}")
            await asyncio.sleep(86400) # 24 hours
            
    asyncio.create_task(daily_scraper_task())

# ===============================
# Live Vegetable Prices Endpoint
# ===============================
@app.get("/api/vegetables/recent")
def get_recent_vegetables():
    conn = get_db_connection()
    latest_date_row = conn.execute("SELECT MAX(arrival_date) FROM market_data WHERE market='vegetablemarketprice.com'").fetchone()
    if not latest_date_row or not latest_date_row[0]:
        conn.close()
        return {"status": "success", "data": []}
        
    latest_date = latest_date_row[0]
    
    # Common vegetables to highlight on the dashboard
    targets = ['Tomato (தக்காளி)', 'Onion Big (பெரிய வெங்காயம்)', 'Potato (உருளைக்கிழங்கு)', 'Carrot (கேரட்)', 'Green Chilli (பச்சை மிளகாய்)', 'Cabbage (முட்டைக்கோஸ்)']
    placeholders = ','.join('?' * len(targets))
    
    query = f"SELECT commodity, modal_price, variety FROM market_data WHERE market='vegetablemarketprice.com' AND arrival_date=? AND commodity IN ({placeholders})"
    
    rows = conn.execute(query, [latest_date] + targets).fetchall()
    conn.close()
    
    result = []
    for r in rows:
        result.append({
            "name": r["commodity"].split('(')[0].strip(),
            "price": r["modal_price"],
            "unit": r["variety"]
        })
        
    return {"status": "success", "date": latest_date, "data": result}

@app.get("/api/vegetables/all_recent")
def get_all_recent_vegetables(date: str = Query(None, description="Format YYYY-MM-DD")):
    conn = get_db_connection()
    if date:
        latest_date = date
    else:
        latest_date_row = conn.execute("SELECT MAX(arrival_date) FROM market_data WHERE market='vegetablemarketprice.com'").fetchone()
        if not latest_date_row or not latest_date_row[0]:
            conn.close()
            return {"status": "success", "data": []}
            
        latest_date = latest_date_row[0]
        
    # Calculate 10 days ago for average calculation
    from datetime import datetime, timedelta
    ld_obj = datetime.strptime(latest_date, "%Y-%m-%d")
    ten_days_ago_str = (ld_obj - timedelta(days=10)).strftime("%Y-%m-%d")
    
    # Get averages over the last 10 days
    avg_query = """
        SELECT commodity, AVG(modal_price) as avg_price
        FROM market_data
        WHERE market='vegetablemarketprice.com' AND arrival_date > ? AND arrival_date <= ?
        GROUP BY commodity
    """
    avg_rows = conn.execute(avg_query, [ten_days_ago_str, latest_date]).fetchall()
    
    averages = {}
    for r in avg_rows:
        averages[r["commodity"]] = round(r["avg_price"], 2)
    
    query = "SELECT commodity, min_price, max_price, modal_price, variety FROM market_data WHERE market='vegetablemarketprice.com' AND arrival_date=? ORDER BY commodity ASC"
    rows = conn.execute(query, [latest_date]).fetchall()
    conn.close()
    
    result = []
    for r in rows:
        commodity_full = r["commodity"]
        name_en = commodity_full.split('(')[0].strip() if '(' in commodity_full else commodity_full
        name_ta = commodity_full.split('(')[1].replace(')', '').strip() if '(' in commodity_full else ""
        
        result.append({
            "name": name_en,
            "name_ta": name_ta,
            "min_price": r["min_price"],
            "max_price": r["max_price"],
            "price": r["modal_price"],
            "avg_10_days": averages.get(commodity_full, r["modal_price"]),
            "unit": r["variety"]
        })
        
    return {"status": "success", "date": latest_date, "data": result}
        
@app.get("/api/market_trends")
def get_market_trends(date: str = Query(None, description="Format YYYY-MM-DD")):
    conn = get_db_connection()
    from datetime import datetime
    
    if date:
        target_date = date
    else:
        target_date = datetime.now().strftime("%Y-%m-%d")
    
    rows = conn.execute("SELECT commodity, trend_type, trend_value FROM market_trends WHERE date=? ORDER BY trend_value DESC", (target_date,)).fetchall()
    
    latest_date = target_date
    if not rows and not date:
        latest_date_row = conn.execute("SELECT MAX(date) FROM market_trends").fetchone()
        if latest_date_row and latest_date_row[0]:
            latest_date = latest_date_row[0]
            rows = conn.execute("SELECT commodity, trend_type, trend_value FROM market_trends WHERE date=? ORDER BY trend_value DESC", (latest_date,)).fetchall()
            
    conn.close()
            
    rises = []
    falls = []
    for r in rows:
        item = {
            "name": r["commodity"],
            "trend": r["trend_value"]
        }
        if r["trend_type"] == "RISE":
            rises.append(item)
        elif r["trend_type"] == "FALL":
            falls.append(item)
            
    return {"status": "success", "date": latest_date, "rises": rises, "falls": falls}

@app.get("/api/market_history")
def get_market_history(commodity: str):
    conn = get_db_connection()
    query = """
    SELECT arrival_date, modal_price 
    FROM market_data 
    WHERE market='vegetablemarketprice.com' AND commodity LIKE ? 
    ORDER BY arrival_date ASC
    """
    rows = conn.execute(query, [f"{commodity}%"]).fetchall()
    conn.close()
    
    dates = []
    prices = []
    for r in rows[-30:]:
        dates.append(r["arrival_date"])
        prices.append(r["modal_price"])
        
    return {"status": "success", "commodity": commodity, "dates": dates, "prices": prices}





# ===============================
# Get Latest Telemetry Data
# ===============================
@app.get("/latest")
def latest_telemetry():
    conn = get_db_connection()
    latest = conn.execute("SELECT * FROM telemetry_data ORDER BY timestamp DESC LIMIT 1").fetchone()
    conn.close()

    if not latest:
        raise HTTPException(status_code=400, detail="Telemetry database is empty.")

    def safe_json_value(val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return val
        return val

    result = {key: safe_json_value(latest[key]) for key in latest.keys()}
    return result

# ===============================
# Crop Recommendation Endpoint
# ===============================
@app.get("/predict")
def predict_crop():
    conn = get_db_connection()
    latest = conn.execute("SELECT * FROM telemetry_data ORDER BY timestamp DESC LIMIT 1").fetchone()
    conn.close()

    if not latest:
        raise HTTPException(status_code=400, detail="Telemetry database is empty.")

    def safe_value(val):
        return float(val) if val is not None else 0.0

    N = safe_value(latest["N"])
    P = safe_value(latest["P"])
    K = safe_value(latest["K"])
    temperature = safe_value(latest["temperature"])
    humidity = safe_value(latest["humidity"])
    ph = safe_value(latest["ph"])
    rainfall = safe_value(latest["rainfall"])
    soil_moisture = safe_value(latest["soil_moisture"])

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    try:
        prediction = model.predict(features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    telemetry_dict = {
        "timestamp": latest["timestamp"],
        "device_id": latest["device_id"],
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
# Weather Endpoint
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
# Irrigation Alert Endpoint
# ===============================
@app.get("/irrigation-alert")
def irrigation_alert():
    conn = get_db_connection()
    latest = conn.execute("SELECT * FROM telemetry_data ORDER BY timestamp DESC LIMIT 1").fetchone()
    conn.close()

    if not latest:
        raise HTTPException(status_code=400, detail="Telemetry database is empty.")
    
    soil_moisture = float(latest["soil_moisture"] or 0.0)

    weather = get_weather()
    rainfall = weather.get("rainfall", 0.0)

    alert_level = "normal"
    if soil_moisture < 30 and rainfall < 2:
        alert_message = "Low soil moisture and no rain - Irrigation needed immediately!"
        alert_level = "critical"
    elif soil_moisture < 30 and rainfall >= 2:
        alert_message = "Low soil moisture but rain expected - Irrigation can be delayed."
        alert_level = "warning"
    else:
        alert_message = "Soil moisture is sufficient - No irrigation needed."
        alert_level = "normal"

    return {
        "soil_moisture": soil_moisture,
        "rainfall": rainfall,
        "alert_message": alert_message,
        "alert_level": alert_level
    }


# ===============================
# Crop Yield Prediction Endpoint
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
# Auto-fill Endpoint for yield.html
# ===============================
@app.get("/latest-sensor-data")
def latest_sensor_data():
    conn = get_db_connection()
    latest = conn.execute("SELECT * FROM telemetry_data ORDER BY timestamp DESC LIMIT 1").fetchone()
    conn.close()

    if not latest:
        raise HTTPException(status_code=400, detail="Telemetry database is empty.")
    
    def safe_value(val):
        return float(val) if val is not None else 0.0

    N = safe_value(latest["N"])
    P = safe_value(latest["P"])
    K = safe_value(latest["K"])
    temperature = safe_value(latest["temperature"])
    humidity = safe_value(latest["humidity"])
    ph = safe_value(latest["ph"])
    rainfall = safe_value(latest["rainfall"])

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

# ===============================
# MARKET INSIGHTS - USING SQLITE
# ===============================

@app.get("/market_meta")
def market_meta():
    """
    Lightweight endpoint to populate dropdowns quickly (states, districts, commodities, date range).
    """
    conn = get_db_connection()
    
    # Get all State-District pairs
    pairs = conn.execute("SELECT DISTINCT state, district FROM market_data WHERE state IS NOT NULL AND district IS NOT NULL ORDER BY state, district").fetchall()
    
    # Build Map and flat lists
    state_district_map = {}
    states = set()
    districts = set()
    
    for r in pairs:
        s, d = r[0], r[1]
        states.add(s)
        districts.add(d)
        if s not in state_district_map:
            state_district_map[s] = []
        state_district_map[s].append(d)
        
    commodities = [row[0] for row in conn.execute("SELECT DISTINCT commodity FROM market_data WHERE commodity IS NOT NULL ORDER BY commodity").fetchall()]
    
    date_min_row = conn.execute("SELECT MIN(arrival_date) FROM market_data WHERE arrival_date IS NOT NULL").fetchone()
    date_max_row = conn.execute("SELECT MAX(arrival_date) FROM market_data WHERE arrival_date IS NOT NULL").fetchone()
    
    date_min = date_min_row[0] if date_min_row and date_min_row[0] else ""
    date_max = date_max_row[0] if date_max_row and date_max_row[0] else ""
    
    total = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
    conn.close()

    return {
        "status": "success",
        "total": total,
        "states": sorted(list(states)),
        "districts": sorted(list(districts)),
        "state_district_map": state_district_map, 
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
    conn = get_db_connection()
    
    query = "SELECT * FROM market_data WHERE 1=1"
    params = []

    if state and state.upper() != "ALL":
        query += " AND state LIKE ?"
        params.append(state)
    if district and district.upper() != "ALL":
        query += " AND district LIKE ?"
        params.append(district)
    if commodity and commodity.upper() != "ALL":
        query += " AND commodity LIKE ?"
        params.append(commodity)
    
    if date_from:
        query += " AND arrival_date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND arrival_date <= ?"
        params.append(date_to)
        
    # Count total
    count_query = query.replace("SELECT *", "SELECT COUNT(*)")
    total = conn.execute(count_query, params).fetchone()[0]
    
    # Pagination
    query += " ORDER BY arrival_date ASC LIMIT ? OFFSET ?"
    params.append(page_size)
    params.append((page - 1) * page_size)
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    result_rows = [dict(row) for row in rows]

    return {
        "status": "success",
        "source": "sqlite",
        "page": page,
        "page_size": page_size,
        "total": total,
        "columns": ["id", "state", "district", "market", "commodity", "variety", "grade", "arrival_date", "min_price", "max_price", "modal_price"],
        "rows": result_rows,
    }

# ===============================
# Soil Data Endpoint
# ===============================
@app.get("/soil-data")
def get_soil_data(lat: float, lon: float):
    try:
        url = f"http://api.agromonitoring.com/agro/1.0/soil?lat={lat}&lon={lon}&appid={AGRO_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch soil data from Agromonitoring API")

        # The API returns data like:
        # {
        #     "dt": 1586468027,
        #     "t0": 279.85,  # Surface temperature (Kelvin)
        #     "t10": 280.15, # Temperature at 10cm depth
        #     "moisture": 0.169, # Soil moisture m3/m3
        #     "t0_unit": "K",
        #     "t10_unit": "K",
        #     "moisture_unit": "m3/m3"
        # }
        
        # Convert Kelvin to Celsius
        surface_temp = round(data.get("t0", 0) - 273.15, 1)
        temp_10cm = round(data.get("t10", 0) - 273.15, 1)

        # -----------------------------
        # SIMULATE SOIL TYPE (External API doesn't provide it)
        # -----------------------------
        import random
        soil_types_list = [
            "Alluvial Soil", "Black Soil", "Red Soil", "Laterite Soil",
            "Arid Soil", "Forest Soil", "Loamy Soil", "Clay Soil"
        ]
        simulated_soil_type = random.choice(soil_types_list)
        
        return {
            "surface_temp": surface_temp,
            "temp_10cm": temp_10cm,
            "moisture": data.get("moisture", 0),
            "soil_type": simulated_soil_type,
            "original_data": data
        }
    except Exception as e:
        print(f"Error fetching soil data: {e}")
        # Return mock data if API fails (for demo reliability)
        import random
        return {
            "surface_temp": round(random.uniform(20, 35), 1),
            "temp_10cm": round(random.uniform(18, 30), 1),
            "moisture": round(random.uniform(0.1, 0.4), 3),
            "soil_type": random.choice(["Alluvial Soil", "Red Soil", "Black Soil"]),
            "note": "Mock data (API request failed)"
        }

# ===============================
# AI Yield Forecasting Endpoint
# ===============================
@app.get("/yield-predict-ai")
def yield_predict_ai(
    crop: str = Query("Rice"),
    state: str = Query("Tamil Nadu"),
    season: str = Query("Kharif"),
    area: float = Query(1.0),
    rainfall: float = Query(None),
    fertilizer: float = Query(None),
    pesticide: float = Query(None),
):
    # Auto-fill missing params from DB if possible
    if rainfall is None or fertilizer is None:
        conn = get_db_connection()
        latest = conn.execute("SELECT * FROM telemetry_data ORDER BY timestamp DESC LIMIT 1").fetchone()
        conn.close()
        
        if latest:
            if rainfall is None: rainfall = float(latest["rainfall"] or 1000.0)
            # Simple heuristic for fertilizer if not tracked directly in raw kg
            if fertilizer is None: fertilizer = 5000.0 # default placeholder
    
    # Defaults if still None
    rainfall = rainfall or 1000.0
    fertilizer = fertilizer or 5000.0
    pesticide = pesticide or 50.0 # placeholder
    
    features = {
        "Crop": crop,
        "State": state,
        "Season": season,
        "Area": area,
        "Annual_Rainfall": rainfall,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide
    }
    
    result = yield_service_ai.predict_yield(features)
    
    # Add production estimate (Yield * Area)
    if "expected_yield" in result:
        result["expected_production"] = round(result["expected_yield"] * area, 2)
        
    return result

# ===============================
# Decision Engine Endpoint
# ===============================
class ContextRequest(BaseModel):
    growth_stage: str = "Vegetative"
    days_since_last_fert: int = 10
    crop_type: str = "Rice"

@app.post("/recommendations")
def get_recommendations(context: ContextRequest):
    # Fetch latest telemetry
    conn = get_db_connection()
    latest = conn.execute("SELECT * FROM telemetry_data ORDER BY timestamp DESC LIMIT 1").fetchone()
    conn.close()
    
    if not latest:
         telemetry = {"soil_moisture": 50.0, "temperature": 28.0, "rainfall": 0.0, "rainfall_forecast": 0.0}
    else:
        telemetry = {
            "soil_moisture": latest["soil_moisture"],
            "temperature": latest["temperature"],
            "rainfall": latest["rainfall"],
            "rainfall_forecast": 0.0, # Placeholder, in real app fetch from OpenWeather
            "humidity": latest["humidity"]
        }
        
    recs = decision_engine.elaborate_recommendations(
        telemetry, 
        context.dict()
    )
    
    return {"recommendations": recs}

class ChatRequest(BaseModel):
    message: str
    language: str = "en"

@app.post("/api/chat")
def chat_with_rag(request: ChatRequest):
    if rag_service_instance is None:
        return {"reply": "Chatbot is temporarily disabled due to missing dependencies on this python version."}
    try:
        reply = rag_service_instance.query(request.message)
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"Sorry, the chatbot encountered an error: {str(e)}"}

# ===============================
# What-If Simulation Endpoint
# ===============================
from pydantic import BaseModel

class SimulationRequest(BaseModel):
    irrigation_percent: float = 0.0
    fertilizer_percent: float = 0.0
    temp_change: float = 0.0
    rainfall_change: float = 0.0

@app.post("/simulate")
def run_simulation(request: SimulationRequest):
    conn = get_db_connection()
    latest = conn.execute("SELECT * FROM telemetry_data ORDER BY timestamp DESC LIMIT 1").fetchone()
    conn.close()

    if not latest:
        # Fallback if no db data
        baseline = {"soil_moisture": 30.0, "rainfall": 0.0, "temperature": 25.0, "humidity": 60.0}
    else:
        baseline = {
            "soil_moisture": latest["soil_moisture"],
            "rainfall": latest["rainfall"],
            "temperature": latest["temperature"],
            "humidity": latest["humidity"]
        }
    
    # We call our simulation engine
    adjustments = {
        "irrigation_percent": request.irrigation_percent,
        "fertilizer_percent": request.fertilizer_percent,
        "temp_change": request.temp_change
    }
    
    result = sim_engine.run_simulation(baseline, adjustments)
    
    # Run a baseline sim (0 adjustments) to get comparative numbers if needed, 
    # but the service already returns some "delta".
    # For UI clarity, we return both "Simulated" and "Baseline" explicitly.
    baseline_result = sim_engine.run_simulation(baseline, {})
    
    return {
        "baseline": baseline_result,
        "simulated": result
    }

# ===============================
# Farm Profit Calculator Endpoints
# ===============================
class ProfitCalcRequest(BaseModel):
    crop: str
    area: float
    seed_cost: float
    fertilizer_cost: float
    pesticide_cost: float
    irrigation_cost: float
    labor_cost: float
    machinery_cost: float
    misc_cost: float
    expected_yield: float
    market_price: float

@app.post("/api/profit/calculate")
def calculate_profit(request: ProfitCalcRequest):
    total_cost = (
        request.seed_cost + request.fertilizer_cost + request.pesticide_cost +
        request.irrigation_cost + request.labor_cost + request.machinery_cost + request.misc_cost
    )
    total_production = request.area * request.expected_yield
    estimated_revenue = total_production * request.market_price
    net_profit = estimated_revenue - total_cost

    recommendations = []
    if total_cost > 0:
        if request.fertilizer_cost > total_cost * 0.2:
            recommendations.append("Fertilizer cost is high (>20%). Consider soil testing for optimized application.")
        if request.labor_cost > total_cost * 0.3:
            recommendations.append("Labor costs are high. Explore mechanization tools for your crop.")
    
    if net_profit < 0:
        recommendations.append("Warning: This scenario results in a loss. Review input costs or hold harvest for better market prices.")
    elif net_profit > 0 and total_cost > 0 and (net_profit / total_cost) > 1.0:
        recommendations.append("Excellent projected ROI! Focus on maintaining yield quality.")

    return {
        "status": "success",
        "data": {
            "total_cost": round(total_cost, 2),
            "total_production": round(total_production, 2),
            "estimated_revenue": round(estimated_revenue, 2),
            "net_profit": round(net_profit, 2),
            "recommendations": recommendations
        }
    }

@app.post("/api/profit/history")
def save_profit_history(request: ProfitCalcRequest):
    total_cost = (
        request.seed_cost + request.fertilizer_cost + request.pesticide_cost +
        request.irrigation_cost + request.labor_cost + request.machinery_cost + request.misc_cost
    )
    total_production = request.area * request.expected_yield
    estimated_revenue = total_production * request.market_price
    net_profit = estimated_revenue - total_cost

    conn = get_db_connection()
    conn.execute('''
        INSERT INTO profit_history (
            crop, area, seed_cost, fertilizer_cost, pesticide_cost, irrigation_cost,
            labor_cost, machinery_cost, misc_cost, expected_yield, market_price,
            total_cost, estimated_revenue, net_profit
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        request.crop, request.area, request.seed_cost, request.fertilizer_cost, request.pesticide_cost,
        request.irrigation_cost, request.labor_cost, request.machinery_cost, request.misc_cost,
        request.expected_yield, request.market_price, total_cost, estimated_revenue, net_profit
    ))
    conn.commit()
    inserted_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return {"status": "success", "message": "Scenario saved", "id": inserted_id}

@app.get("/api/profit/history")
def get_profit_history(limit: int = 10):
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM profit_history ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return {"status": "success", "data": [dict(r) for r in rows]}

# ===============================
# Telegram Bot Integration 
# ===============================
import sqlite3
class BotRegisterRequest(BaseModel):
    telegram_id: str
    name: str

@app.post("/api/bot/register")
def register_bot_user(req: BotRegisterRequest):
    conn = get_db_connection()
    try:
        conn.execute("INSERT INTO bot_users (telegram_id, name) VALUES (?, ?)", (req.telegram_id, req.name))
        conn.commit()
    except sqlite3.IntegrityError:
        pass # Already registered
    finally:
        conn.close()
    return {"status": "success", "message": "User registered"}

@app.get("/api/tasks")
def get_bot_tasks(telegram_id: str = None):
    conn = get_db_connection()
    query = "SELECT * FROM farm_tasks WHERE status = 'PENDING'"
    params = []
    if telegram_id:
        query += " AND user_id = ?"
        params.append(telegram_id)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return {"status": "success", "data": [dict(r) for r in rows]}

@app.post("/api/tasks/{task_id}/complete")
def complete_bot_task(task_id: int):
    conn = get_db_connection()
    conn.execute("UPDATE farm_tasks SET status = 'COMPLETED' WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
    return {"status": "success", "message": "Task completed"}

@app.get("/api/weather/current")
def get_current_weather():
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={OPENWEATHER_API_KEY}&units={UNITS}"
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            return {
                "status": "success",
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "condition": data["weather"][0]["description"].title(),
                "rain_prob": data.get("clouds", {}).get("all", 0) # approximation
            }
    except:
        pass
    return {"status": "error", "temp": 28.0, "humidity": 65, "condition": "Sunny (Mock)", "rain_prob": 10}

@app.get("/api/profit/summary")
def get_profit_summary():
    conn = get_db_connection()
    row = conn.execute("SELECT total_cost, estimated_revenue, net_profit FROM profit_history ORDER BY created_at DESC LIMIT 1").fetchone()
    conn.close()
    if row:
        return {"status": "success", "data": dict(row)}
    return {"status": "success", "data": {"total_cost": 0, "estimated_revenue": 0, "net_profit": 0}}

# ===============================
# Serve Frontend
# ===============================
# Mount requests to / to the frontend folder
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "..", "frontend"), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
