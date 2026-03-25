from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)

def test_endpoints():
    print("Testing /api/bot/register...")
    response = client.post("/api/bot/register", json={"telegram_id": "123456789", "name": "Fake User"})
    assert response.status_code == 200
    print("[PASS] /api/bot/register OK")

    print("Testing /api/tasks...")
    response = client.get("/api/tasks?telegram_id=123456789")
    assert response.status_code == 200
    print("[PASS] /api/tasks OK", response.json())

    print("Testing /api/weather/current...")
    response = client.get("/api/weather/current")
    assert response.status_code == 200
    print("[PASS] /api/weather/current OK", response.json())

    print("Testing /api/profit/summary...")
    response = client.get("/api/profit/summary")
    assert response.status_code == 200
    print("[PASS] /api/profit/summary OK", response.json())

if __name__ == "__main__":
    test_endpoints()
    print("\nAll endpoints tested successfully!")
