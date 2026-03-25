import os
import requests
import sqlite3
import time
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "agritwin.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_all_veg_ids():
    url = "https://vegetablemarketprice.com/api/dataapi/market/tamilnadu/chartdatavalues?start=2024-05-10&end=2024-05-15&vegIds=1"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        data = r.json()
        return [item['vegId'] for item in data.get('columnMapping', [])]
    except Exception as e:
        print(f"Error fetching veg_ids: {e}")
        return []

def backfill_history(days=30):
    print(f"[{datetime.now()}] Starting historical backfill for {days} days...")
    veg_ids = get_all_veg_ids()
    if not veg_ids:
        print("Could not retrieve veg targets.")
        return
        
    print(f"Found {len(veg_ids)} vegetables to backfill.")
        
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    chunk_size = 8  # API seems to handle 8 items nicely
    
    conn = get_db_connection()
    cursor = conn.cursor()
    count = 0
    
    for i in range(0, len(veg_ids), chunk_size):
        chunk = veg_ids[i:i+chunk_size]
        ids_str = ",".join(chunk)
        url = f"https://vegetablemarketprice.com/api/dataapi/market/tamilnadu/chartdatavalues?start={start_str}&end={end_str}&vegIds={ids_str}"
        print(f"Fetching chunk {i//chunk_size + 1}/{len(veg_ids)//chunk_size + 1}...")
        
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            data = r.json()
            
            columns = data.get('columns', []) 
            series_list = data.get('data', [])
            
            for series in series_list:
                commodity_full = series.get('name', '')
                commodity = commodity_full.split('(')[0].strip() if '(' in commodity_full else commodity_full
                
                data_points = series.get('data', [])
                
                for idx, pt in enumerate(data_points):
                    if not pt or 'y' not in pt or idx >= len(columns):
                        continue
                        
                    date_val = columns[idx]
                    modal_price = float(pt.get('y', 0))
                    min_p = modal_price
                    max_p = modal_price
                    
                    retail_str = pt.get('retailprice', '')
                    if retail_str and '-' in retail_str:
                        try:
                            parts = retail_str.split('-')
                            min_p = float(parts[0].strip())
                            max_p = float(parts[1].strip())
                        except:
                            pass
                            
                    variety = pt.get('units', '1kg')
                    
                    # insert
                    exists = cursor.execute(
                        "SELECT id FROM market_data WHERE market=? AND commodity=? AND arrival_date=?",
                        ("vegetablemarketprice.com", commodity, date_val)
                    ).fetchone()
                    
                    if not exists:
                        cursor.execute('''
                            INSERT INTO market_data (state, district, market, commodity, variety, grade, arrival_date, min_price, max_price, modal_price)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            "Tamil Nadu", "ALL", "vegetablemarketprice.com", 
                            commodity, variety, "FAQ", 
                            date_val, 
                            min_p, max_p, modal_price
                        ))
                        count += 1
            
            time.sleep(1) # delay to be nice to API
                        
        except Exception as e:
            print(f"Error processing chunk: {e}")
            
    conn.commit()
    conn.close()
    print(f"[{datetime.now()}] Backfill complete! Inserted {count} historical records.")

if __name__ == "__main__":
    backfill_history(30)
