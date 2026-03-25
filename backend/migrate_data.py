import pandas as pd
import os
from database import get_db_connection, init_db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def migrate_telemetry():
    telemetry_path = os.path.join(BASE_DIR, "telemetry.csv")
    if not os.path.exists(telemetry_path):
        print("⚠️ telemetry.csv not found. Skipping telemetry migration.")
        return

    print("Migrating telemetry.csv...")
    try:
        df = pd.read_csv(telemetry_path, on_bad_lines="skip")
        conn = get_db_connection()
        cursor = conn.cursor()

        count = 0
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO telemetry_data (timestamp, device_id, N, P, K, temperature, humidity, ph, rainfall, soil_moisture)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row.get("timestamp"),
                row.get("device_id"),
                row.get("N"),
                row.get("P"),
                row.get("K"),
                row.get("temperature"),
                row.get("humidity"),
                row.get("ph"),
                row.get("rainfall"),
                row.get("soil_moisture")
            ))
            count += 1
        
        conn.commit()
        conn.close()
        print(f"Migrated {count} rows to telemetry_data.")
    except Exception as e:
        print(f"Error migrating telemetry: {e}")

def migrate_market():
    market_path = os.path.join(BASE_DIR, "..", "dataset", "market.csv")
    if not os.path.exists(market_path):
        print(f"Market.csv not found at {market_path}. Skipping market migration.")
        return

    print("Migrating market.csv...")
    try:
        df = pd.read_csv(market_path)
        
        # Renaissance columns to match DB
        # CSV Cols: State, District, Market, Commodity, Variety, Grade, Arrival_Date, Min_x0020_Price, Max_x0020_Price, Modal_x0020_Price
        df = df.rename(columns={
            "Min_x0020_Price": "min_price",
            "Max_x0020_Price": "max_price",
            "Modal_x0020_Price": "modal_price",
            "State": "state",
            "District": "district",
            "Market": "market",
            "Commodity": "commodity",
            "Variety": "variety",
            "Grade": "grade",
            "Arrival_Date": "arrival_date"
        })

        conn = get_db_connection()
        cursor = conn.cursor()

        count = 0
        for _, row in df.iterrows():
            # Basic validation
            if pd.isna(row.get("modal_price")):
                continue

            cursor.execute('''
                INSERT INTO market_data (state, district, market, commodity, variety, grade, arrival_date, min_price, max_price, modal_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row.get("state"),
                row.get("district"),
                row.get("market"),
                row.get("commodity"),
                row.get("variety"),
                row.get("grade"),
                row.get("arrival_date"),
                row.get("min_price"),
                row.get("max_price"),
                row.get("modal_price")
            ))
            count += 1
            
        conn.commit()
        conn.close()
        print(f"Migrated {count} rows to market_data.")

    except Exception as e:
        print(f"Error migrating market data: {e}")

if __name__ == "__main__":
    init_db()
    migrate_telemetry()
    migrate_market()
