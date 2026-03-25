import sqlite3
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "agritwin.db")

def inspect_db():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    print(f"Inspection: {DB_PATH}\n")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    if not tables:
        print("Empty database. No tables found.")
        return

    for table_name in tables:
        table = table_name[0]
        print(f"=== Table: {table} ===")
        
        # Get count
        count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"Count: {count} rows")
        
        # Get columns
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [info[1] for info in cursor.fetchall()]
        print(f"Columns: {columns}")

        # Show head
        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn)
        if not df.empty:
            print("\nHead (5 rows):")
            print(df.to_string(index=False))
        else:
            print("\n(Table is empty)")
        print("\n" + "-"*40 + "\n")

    conn.close()

if __name__ == "__main__":
    inspect_db()
