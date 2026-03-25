import sqlite3
import os

DB_PATH = os.path.join("backend", "agritwin.db")

def check_dates():
    if not os.path.exists(DB_PATH):
        print(f"DB not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("--- Sample Dates ---")
    rows = cursor.execute("SELECT arrival_date FROM market_data LIMIT 10").fetchall()
    for r in rows:
        print(r[0])
        
    print("\n--- Min/Max Date (String Sort) ---")
    min_d = cursor.execute("SELECT MIN(arrival_date) FROM market_data").fetchone()[0]
    max_d = cursor.execute("SELECT MAX(arrival_date) FROM market_data").fetchone()[0]
    print(f"Min: {min_d}")
    print(f"Max: {max_d}")

    conn.close()

if __name__ == "__main__":
    check_dates()
