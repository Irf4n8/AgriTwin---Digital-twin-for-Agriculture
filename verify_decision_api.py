import requests
import json
import time

URL = "http://127.0.0.1:8000/recommendations"

def test_decision_api():
    print("Testing Decision Engine API Endpoint...")
    
    # Wait for server
    time.sleep(5) 
    
    # Test Payload
    payload = {
        "growth_stage": "Vegetative",
        "days_since_last_fert": 20,
        "crop_type": "Rice"
    }
    
    try:
        res = requests.post(URL, json=payload)
        if res.status_code == 200:
            data = res.json()
            recs = data.get("recommendations", [])
            print(f"\nReceived {len(recs)} recommendations.")
            
            for r in recs:
                print(f"[{r.get('source', 'Unknown')}] {r.get('action')}")
                print(f"   Reason: {r.get('reason')}")
                print(f"   Benefits: {r.get('benefit')}")
                print("-" * 30)
                
            if len(recs) > 0:
                print("PASS: Recommendations received.")
            else:
                print("info: No recommendations triggered (might be normal if mock data is optimal). check database/telemetry.")
                
        else:
            print(f"FAIL: HTTP {res.status_code}")
            print(res.text)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_decision_api()
