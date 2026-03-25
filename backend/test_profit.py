from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_profit_calculator():
    payload = {
        "crop": "Tomato",
        "area": 10,
        "seed_cost": 5000,
        "fertilizer_cost": 15000,
        "pesticide_cost": 5000,
        "irrigation_cost": 5000,
        "labor_cost": 20000,
        "machinery_cost": 10000,
        "misc_cost": 5000,
        "expected_yield": 4500,
        "market_price": 20
    }

    # Test Calculation
    print("Testing /api/profit/calculate...")
    response = client.post("/api/profit/calculate", json=payload)
    assert response.status_code == 200
    data = response.json()
    print("Calculation Result:", data)

    if data["status"] == "success":
        print("✅ Calculation OK")
    
    cost = data["data"]["total_cost"]
    assert cost == sum([5000, 15000, 5000, 5000, 20000, 10000, 5000])

    # Test Save History
    print("\nTesting /api/profit/history (POST)...")
    response_history = client.post("/api/profit/history", json=payload)
    assert response_history.status_code == 200
    print("Save History Result:", response_history.json())
    print("✅ Save OK")

    # Test Get History
    print("\nTesting /api/profit/history (GET)...")
    get_res = client.get("/api/profit/history")
    assert get_res.status_code == 200
    history_data = get_res.json()
    print("Got History records count:", len(history_data.get("data", [])))
    print("✅ Get OK")

if __name__ == "__main__":
    test_profit_calculator()
    print("\nAll backend profit endpoints pass!")
