import requests
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../dataset")

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# URLs
CROP_REC_URL = "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv"
CROP_YIELD_URL_1 = "https://raw.githubusercontent.com/MarwanMusa/Crop-Yield-Prediction/main/crop_yield.csv"
CROP_YIELD_URL_2 = "https://raw.githubusercontent.com/sahil0701/Crop-Yield-Prediction/master/crop_yield.csv"

def download_file(url, dest_path):
    print(f"Downloading from {url}...")
    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
        print(f"Saved to {dest_path}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

# 1. Crop Recommendation
rec_path = os.path.join(DATASET_DIR, "Crop_recommendation.csv")
if download_file(CROP_REC_URL, rec_path):
    df = pd.read_csv(rec_path)
    print(f"Crop Recommendation Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

# 2. Crop Yield
yield_path = os.path.join(DATASET_DIR, "crop_yield.csv")
success = download_file(CROP_YIELD_URL_1, yield_path)
if not success:
    print("Trying backup URL for Crop Yield...")
    success = download_file(CROP_YIELD_URL_2, yield_path)

if success:
    try:
        df = pd.read_csv(yield_path)
        print(f"Crop Yield Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading crop_yield.csv: {e}")
