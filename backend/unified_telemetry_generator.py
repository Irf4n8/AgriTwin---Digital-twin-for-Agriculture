import csv
import random
import time
from datetime import datetime
import os
from database import get_db_connection, init_db


# ===============================
# 🪴 Arduino Serial Setup
# ===============================
try:
    import serial
    ARDUINO_AVAILABLE = True
except ImportError:
    ARDUINO_AVAILABLE = False

ARDUINO_AVAILABLE = False  # ⚠️ FORCED SIMULATION MODE by default

ARDUINO_PORT = "COM07"  # ⚠️ Change this to your correct port (e.g., COM3, COM5)
BAUD_RATE = 9600
TIMEOUT = 2  # seconds

# ===============================
# 📝 Database Setup
# ===============================
init_db()
print("Initialized database connection")

# ===============================
# 🔌 Connect to Arduino (if available)
# ===============================
ser = None
if ARDUINO_AVAILABLE:
    try:
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=TIMEOUT)
        print(f"Arduino detected on port {ARDUINO_PORT}")
        time.sleep(2)  # wait for Arduino to reset
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
        ser = None
else:
    print("PySerial not installed. Falling back to simulated data.")

print("Telemetry generator started...")

# ===============================
# ♻️ Continuous Data Generation
# ===============================
while True:
    timestamp = datetime.now().isoformat()
    device_id = "device-01"

    # ---------------------------
    # 📡 Real sensor reading
    # ---------------------------
    if ser:
        line = ser.readline().decode("utf-8").strip()
        if line:
            try:
                temp_str, hum_str, soil_str = line.split(",")
                temperature = float(temp_str)
                humidity = float(hum_str)
                soil_moisture = float(soil_str)
            except Exception as e:
                print(f"Invalid Arduino data received: '{line}' | Error: {e}")
                continue
        else:
            print("No data received from Arduino")
            continue
    else:
        # 🧪 Simulated data if Arduino is not available
        temperature = round(random.uniform(15, 40), 1)
        humidity = round(random.uniform(30, 90), 1)
        soil_moisture = round(random.uniform(10, 60), 1)

    # ---------------------------
    # 🎲 Simulated values for NPK, pH, Rainfall
    # ---------------------------
    N = random.randint(0, 140)
    P = random.randint(5, 145)
    K = random.randint(5, 205)
    ph = round(random.uniform(4.5, 8.5), 1)
    rainfall = round(random.uniform(0, 250), 1)

    # ---------------------------
    # 🪄 Append to Database
    # ---------------------------
    try:
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO telemetry_data (timestamp, device_id, N, P, K, temperature, humidity, ph, rainfall, soil_moisture)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, device_id, N, P, K, temperature, humidity, ph, rainfall, soil_moisture))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving to database: {e}")

    # ---------------------------
    # 🖨️ Log to console
    # ---------------------------
    print(
        f"{timestamp} | Temp:{temperature}C Hum:{humidity}% Soil:{soil_moisture}% "
        f"N:{N} P:{P} K:{K} pH:{ph} Rain:{rainfall}"
    )

    time.sleep(3)  # wait before next reading
