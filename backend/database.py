import sqlite3
import os

DB_NAME = "agritwin.db"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, DB_NAME)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    
    # Create Telemetry Data Table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS telemetry_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            soil_moisture REAL,
            temperature REAL,
            humidity REAL,
            ph_level REAL,
            nitrogen REAL,
            phosphorus REAL,
            potassium REAL,
            light_intensity REAL
        )
    ''')

    # Create Market Data Table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state TEXT,
            district TEXT,
            market TEXT,
            commodity TEXT,
            variety TEXT,
            grade TEXT,
            arrival_date TEXT,
            min_price REAL,
            max_price REAL,
            modal_price REAL
        )
    ''')
    
    # Create Market Trends Table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS market_trends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            commodity TEXT,
            trend_type TEXT,
            trend_value REAL
        )
    ''')

    # Create Profit History Table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS profit_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop TEXT,
            area REAL,
            seed_cost REAL,
            fertilizer_cost REAL,
            pesticide_cost REAL,
            irrigation_cost REAL,
            labor_cost REAL,
            machinery_cost REAL,
            misc_cost REAL,
            expected_yield REAL,
            market_price REAL,
            total_cost REAL,
            estimated_revenue REAL,
            net_profit REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create Bot Users Table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS bot_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id TEXT UNIQUE,
            name TEXT,
            connected_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create Farm Tasks Table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS farm_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            task_name TEXT,
            field_zone TEXT,
            due_time DATETIME,
            status TEXT DEFAULT 'PENDING'
        )
    ''')

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

if __name__ == "__main__":
    init_db()
