import requests
import time

URL = "http://127.0.0.1:8000/yield-predict-ai"

def test_yield_ai():
    print("Testing AI Yield Prediction Endpoint...")
    
    # Needs a moment for server to start if just restarted
    time.sleep(10) 
    
    params = {
        "crop": "Rice",
        "state": "Karnataka",
        "season": "Kharif",
        "area": 2.0,
        "rainfall": 1200.0,
        "fertilizer": 50000.0,
        "pesticide": 100.0
    }
    
    try:
        res = requests.get(URL, params=params)
        if res.status_code == 200:
            data = res.json()
            print("\nResponse Received:")
            print(data)
            
            # Validation
            if "expected_yield" in data and "confidence_score" in data:
                print("\nPASS: API returned expected fields.")
                
                # Check Logic
                exp = data['expected_yield']
                best = data['best_case_yield']
                worst = data['worst_case_yield']
                
                if best >= exp >= worst:
                    print("PASS: Logic valid (Best >= Expected >= Worst).")
                else:
                    print(f"FAIL: Logic Error. {best} >= {exp} >= {worst}")
            else:
                print("FAIL: Missing fields.")
        else:
            print(f"FAIL: HTTP {res.status_code}")
            print(res.text)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_yield_ai()
