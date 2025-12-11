import csv
import random
import time
from datetime import datetime
import os

# ===============================
# 🪴 Arduino Serial Setup
# ===============================
try:
    import serial
    ARDUINO_AVAILABLE = True
except ImportError:
    ARDUINO_AVAILABLE = False

ARDUINO_PORT = "COM10"  # ⚠️ Change this to your correct port (e.g., COM3, COM5)
BAUD_RATE = 9600
TIMEOUT = 2  # seconds

# ===============================
# 📝 Telemetry CSV Setup
# ===============================
telemetry_file = "telemetry.csv"
file_exists = os.path.isfile(telemetry_file)

if not file_exists:
    with open(telemetry_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "device_id", "N", "P", "K",
            "temperature", "humidity", "ph", "rainfall", "soil_moisture"
        ])
    print("✅ Created telemetry.csv with headers")

# ===============================
# 🔌 Connect to Arduino (if available)
# ===============================
ser = None
if ARDUINO_AVAILABLE:
    try:
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=TIMEOUT)
        print(f"✅ Arduino detected on port {ARDUINO_PORT}")
        time.sleep(2)  # wait for Arduino to reset
    except Exception as e:
        print(f"❌ Error connecting to Arduino: {e}")
        ser = None
else:
    print("⚠️ PySerial not installed. Falling back to simulated data.")

print("🌾 Telemetry generator started...")

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
                print(f"⚠️ Invalid Arduino data received: '{line}' | Error: {e}")
                continue
        else:
            print("⚠️ No data received from Arduino")
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
    # 🪄 Append to telemetry.csv
    # ---------------------------
    with open(telemetry_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, device_id, N, P, K,
            temperature, humidity, ph, rainfall, soil_moisture
        ])

    # ---------------------------
    # 🖨️ Log to console
    # ---------------------------
    print(
        f"✅ {timestamp} | Temp:{temperature}°C Hum:{humidity}% Soil:{soil_moisture}% "
        f"N:{N} P:{P} K:{K} pH:{ph} Rain:{rainfall}"
    )

    time.sleep(3)  # wait before next reading
