import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join("backend", "agritwin.db")

def fix_dates():
    if not os.path.exists(DB_PATH):
        print(f"DB not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("Fetching rows...")
    rows = cursor.execute("SELECT id, arrival_date FROM market_data").fetchall()
    
    updates = []
    
    for row in rows:
        old_date = row["arrival_date"]
        # Assuming format is DD/MM/YYYY
        try:
            # Parse DD/MM/YYYY
            dt = datetime.strptime(old_date, "%d/%m/%Y")
            new_date = dt.strftime("%Y-%m-%d")
            
            if new_date != old_date:
                updates.append((new_date, row["id"]))
        except ValueError:
            # Maybe already in YYYY-MM-DD or invalid
            pass

    print(f"Updating {len(updates)} rows to YYYY-MM-DD format...")
    
    cursor.executemany("UPDATE market_data SET arrival_date = ? WHERE id = ?", updates)
    
    conn.commit()
    conn.close()
    
    print("Dates fixed.")

if __name__ == "__main__":
    fix_dates()
