import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "agritwin.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def parse_price(price_str):
    # e.g. '₹25 - 32' or '₹21'
    p = price_str.replace('₹', '').strip()
    if not p:
        return 0.0, 0.0
    if '-' in p:
        parts = p.split('-')
        try:
            return float(parts[0].strip()), float(parts[1].strip())
        except:
            pass
    try:
        val = float(p)
        return val, val
    except:
        return 0.0, 0.0

def scrape_market_trends():
    print(f"[{datetime.now()}] Starting market trends scrape...")
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get("https://vegetablemarketprice.com/", headers=headers, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching home page for trends: {e}")
        return False
        
    soup = BeautifulSoup(response.text, 'html.parser')
    today_str = datetime.now().strftime("%Y-%m-%d")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Clear existing trends for today
    cursor.execute("DELETE FROM market_trends WHERE date=?", (today_str,))
    
    rise_body = soup.find('div', class_='price-rise-container-body')
    if rise_body:
        for row in rise_body.find_all('tr'):
            cols = [c.text.strip() for c in row.find_all('td')]
            if len(cols) >= 2:
                try:
                    val = float(cols[1])
                    cursor.execute("INSERT INTO market_trends (date, commodity, trend_type, trend_value) VALUES (?, ?, ?, ?)",
                                 (today_str, cols[0], 'RISE', val))
                except:
                    pass
                    
    fall_body = soup.find('div', class_='price-fall-container-body')
    if fall_body:
        for row in fall_body.find_all('tr'):
            cols = [c.text.strip() for c in row.find_all('td')]
            if len(cols) >= 2:
                try:
                    val = float(cols[1])
                    cursor.execute("INSERT INTO market_trends (date, commodity, trend_type, trend_value) VALUES (?, ?, ?, ?)",
                                 (today_str, cols[0], 'FALL', val))
                except:
                    pass
                    
    conn.commit()
    conn.close()
    print(f"[{datetime.now()}] Finished scraping market trends.")
    return True

def scrape_vegetable_prices(state_url="https://vegetablemarketprice.com/market/tamilnadu/today", state_name="Tamil Nadu"):
    print(f"[{datetime.now()}] Starting vegetable market price scrape for {state_name}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(state_url, headers=headers, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return False

    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all('table')
    if not tables:
        print("No tables found on the page.")
        return False
        
    table = tables[0]
    rows = table.find_all('tr')
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    count = 0
    for row in rows[1:]:  # skip header
        cols = row.find_all(['td', 'th'])
        if len(cols) < 5:
            continue
            
        col_texts = [col.text.strip() for col in cols]
        
        # col_texts = ['', 'Onion Big (பெரிய வெங்காயம்)', '₹21', '₹25 - 32', '1kg']
        commodity_full = col_texts[1]
        
        # Clean commodity name
        commodity = commodity_full.split('(')[0].strip() if '(' in commodity_full else commodity_full
        variety = col_texts[4] # usually unit like 1kg
        
        modal_price, _ = parse_price(col_texts[2])
        min_price, max_price = parse_price(col_texts[3])
        
        if modal_price == 0.0 and min_price == 0.0:
            continue
            
        # Check if already exists for today
        exists = cursor.execute(
            "SELECT id FROM market_data WHERE market=? AND commodity=? AND arrival_date=?",
            ("vegetablemarketprice.com", commodity, today_str)
        ).fetchone()
        
        if not exists:
            cursor.execute('''
                INSERT INTO market_data (state, district, market, commodity, variety, grade, arrival_date, min_price, max_price, modal_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                state_name, "ALL", "vegetablemarketprice.com", 
                commodity, variety, "FAQ", 
                today_str, 
                min_price, max_price, modal_price
            ))
            count += 1
            
    conn.commit()
    conn.close()
    
    print(f"[{datetime.now()}] Scraped and inserted {count} new vegetable prices into database.")
    
    # Also scrape market trends (Price Rise / Price Fall)
    scrape_market_trends()
    
    return True

if __name__ == "__main__":
    scrape_vegetable_prices()
