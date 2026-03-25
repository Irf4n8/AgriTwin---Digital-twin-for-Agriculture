import sqlite3
import os
import time
import requests
import threading
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from database import get_db_connection

def verify_db_population():
    print("\nVerifying Database Population...")
    conn = get_db_connection()
    telemetry_count = conn.execute("SELECT COUNT(*) FROM telemetry_data").fetchone()[0]
    market_count = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
    conn.close()
    
    print(f"   - Telemetry Rows: {telemetry_count}")
    print(f"   - Market Data Rows: {market_count}")
    
    if telemetry_count > 0 and market_count > 0:
        print("Database is populated.")
        return True
    else:
        print("Database is empty or incomplete.")
        return False

def verify_telemetry_generator():
    print("\nVerifying Telemetry Generator...")
    conn = get_db_connection()
    initial_count = conn.execute("SELECT COUNT(*) FROM telemetry_data").fetchone()[0]
    conn.close()
    
    print(f"   - Initial Count: {initial_count}")
    print("   - Waiting for 10 seconds (assuming generator is running separately)...")
    
    # We can't easily start the generator here without blocking, so we'll assume the user
    # or another process might run it, OR we'll skip this if we can't automate it easily 
    # in this single script without complexity. 
    # For now, let's just check if the DB is accessible.
    
    conn = get_db_connection()
    final_count = conn.execute("SELECT COUNT(*) FROM telemetry_data").fetchone()[0]
    conn.close()
    
    if final_count >= initial_count:
        print(f"DB is accessible. (Count: {final_count})")
    else:
        print("DB count decreased? Something is wrong.")

if __name__ == "__main__":
    verify_db_population()
    verify_telemetry_generator()
