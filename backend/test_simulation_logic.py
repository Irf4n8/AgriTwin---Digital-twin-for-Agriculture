import requests
import json

URL = "http://127.0.0.1:8000/simulate"

def test_simulation():
    print("Testing What-If Simulation Endpoint...")
    
    # 1. Baseline Test (No changes)
    print("\n--- Test 1: Zero Adjustments (Baseline) ---")
    payload = {
        "irrigation_percent": 0.0,
        "fertilizer_percent": 0.0,
        "temp_change": 0.0
    }
    
    try:
        res = requests.post(URL, json=payload)
        if res.status_code == 200:
            data = res.json()
            # print(json.dumps(data, indent=2))
            base_yield = data['baseline']['metrics']['yield_kg']
            sim_yield = data['simulated']['metrics']['yield_kg']
            print(f"Success. Baseline Yield: {base_yield}, Sim Yield: {sim_yield}")
            if base_yield == sim_yield:
                print("PASS: Baseline matches Simulation with 0 adjustments.")
            else:
                print("FAIL: Baseline and Sim should match.")
        else:
            print(f"FAIL: HTTP {res.status_code}")
            print(res.text)
            return

    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Positive Adjustment Test
    print("\n--- Test 2: Increase Fertilizer (+20%) ---")
    payload = {
        "irrigation_percent": 0.0,
        "fertilizer_percent": 20.0,
        "temp_change": 0.0
    }
    
    res = requests.post(URL, json=payload)
    data = res.json()
    sim_yield = data['simulated']['metrics']['yield_kg']
    sim_cost = data['simulated']['metrics']['cost']
    base_yield = data['baseline']['metrics']['yield_kg']
    base_cost = data['baseline']['metrics']['cost']
    
    print(f"New Yield: {sim_yield} (Baseline: {base_yield})")
    print(f"New Cost: {sim_cost} (Baseline: {base_cost})")
    
    if sim_yield > base_yield and sim_cost > base_cost:
        print("PASS: Yield and Cost increased as expected.")
    else:
        print("FAIL: Expected increase in yield/cost.")

    # 3. Stress Test (High Temp)
    print("\n--- Test 3: Increase Temp (+5C) ---")
    payload = {
        "irrigation_percent": 0.0,
        "fertilizer_percent": 0.0,
        "temp_change": 5.0
    }
    res = requests.post(URL, json=payload)
    data = res.json()
    stress = data['simulated']['metrics']['stress_index']
    print(f"Stress Index: {stress}%")
    
    if stress > 0:
        print("PASS: Stress increased with temperature.")
    else:
        print("FAIL: Stress should increase.")

if __name__ == "__main__":
    test_simulation()
