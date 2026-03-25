import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from main import get_market_insights, predict_crop, market_meta
    print("Imports successful.")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

print("\n--- Testing Market Meta ---")
try:
    meta = market_meta()
    print(f"Total rows: {meta.get('total')}")
    print(f"States: {len(meta.get('states', []))}")
    if meta.get('total') == 0:
        print("Warning: Total rows is 0!")
    else:
        print("Market Meta OK.")
except Exception as e:
    print(f"Market Meta Failed: {e}")

print("\n--- Testing Market Insights ---")
try:
    # Pass explicit arguments to avoid 'Query' object default issue
    insights = get_market_insights(
        page=1, 
        page_size=10, 
        state="ALL", 
        district="ALL", 
        commodity="ALL",
        date_from=None,
        date_to=None
    )
    print(f"Returned {len(insights.get('rows', []))} rows.")
    if len(insights.get('rows', [])) > 0:
        print(f"Sample Row: {insights['rows'][0]}")
        print("Market Insights OK.")
    else:
        print("Warning: Market Insights returned empty list.")
except Exception as e:
    print(f"Market Insights Failed: {e}")

# Note: predict_crop relies on `latest` telemetry which might be empty if no generator ran.
# But we can check if it runs without crashing.
print("\n--- Testing Crop Prediction ---")
try:
    # This might fail if DB is empty, but we expect it to be populated from migration?
    # No, telemetry migration populated it.
    pred = predict_crop()
    print(f"Prediction result: {pred.get('recommended_crop')}")
    print("Crop Prediction OK.")
except Exception as e:
    print(f"Crop Prediction Failed (Expected if DB empty): {e}")

print("\n--- Done ---")
