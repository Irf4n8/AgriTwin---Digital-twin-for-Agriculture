from decision_engine import DecisionEngine

def test_engine():
    engine = DecisionEngine()
    
    print("--- Test Case 1: Critical Drought ---")
    telemetry_drought = {
        "soil_moisture": 25.0, # < 30 (Trigger Rule)
        "temperature": 32.0,
        "rainfall_forecast": 0.0,
        "rainfall": 0.0,
        "humidity": 40.0
    }
    context = {"growth_stage": "Vegetative", "days_since_last_fert": 5}
    
    recs = engine.elaborate_recommendations(telemetry_drought, context)
    for r in recs:
        print(f"[{r['source']}] {r['action']}: {r['reason']}")
        
    print("\n--- Test Case 2: Optimization Opportunity ---")
    telemetry_opt = {
        "soil_moisture": 55.0, # Decent moisture
        "temperature": 28.0,
        "rainfall_forecast": 0.0,
        "rainfall": 0.0,
        "humidity": 60.0
    }
    # Pretend growth stage allows fert
    context_opt = {"growth_stage": "Vegetative", "days_since_last_fert": 20} # > 14 (Trigger Rule)
    
    recs = engine.elaborate_recommendations(telemetry_opt, context_opt)
    for r in recs:
        print(f"[{r['source']}] {r['action']}: {r['reason']}")

if __name__ == "__main__":
    test_engine()
