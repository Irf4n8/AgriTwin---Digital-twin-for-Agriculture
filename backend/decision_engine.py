import json
import os
from simulation_service import AgriSimulationEngine

class DecisionEngine:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.rules_path = os.path.join(self.base_dir, "decision_rules.json")
        self.rules = self.load_rules()
        self.sim_engine = AgriSimulationEngine()

    def load_rules(self):
        try:
            with open(self.rules_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading rules: {e}")
            return {}

    def elaborate_recommendations(self, telemetry, crop_context):
        """
        Main function to generate recommendations.
        telemetry: dict {soil_moisture, temperature, rainfall_forecast, ...}
        crop_context: dict {growth_stage, days_since_last_fert, crop_type}
        """
        recommendations = []
        
        # 1. Static Rule Evaluation
        recommendations.extend(self.evaluate_rules(telemetry, crop_context))
        
        # 2. AI Optimization (Dynamic Simulation)
        ai_recs = self.run_ai_optimization(telemetry)
        recommendations.extend(ai_recs)
        
        # Sort by priority (if available) or risk
        # Simple sort: High Risk first
        recommendations.sort(key=lambda x: 0 if x['risk'] == 'High' else 1)
        
        return recommendations

    def evaluate_rules(self, telemetry, context):
        recs = []
        
        # Combine data for eval context
        data = {**telemetry, **context}
        
        for category, rule_list in self.rules.items():
            for rule in rule_list:
                try:
                    # Safe eval: simplified context
                    condition = rule["condition"]
                    if eval(condition, {"__builtins__": None}, data):
                        recs.append({
                            "type": category.capitalize(),
                            "action": rule["action"],
                            "reason": rule["reason"],
                            "benefit": rule["benefit"],
                            "risk": rule["risk"],
                            "source": "Rule-Based"
                        })
                except Exception as e:
                    print(f"Rule Eval Error '{rule['condition']}': {e}")
        return recs

    def run_ai_optimization(self, telemetry):
        """
        Run What-If scenarios to find hidden opportunities.
        """
        recs = []
        
        # Scenario A: Irrigation Optimization
        # If moisture is decent (e.g. 40-60), check if adding more water significantly boosts yield
        if 40 <= telemetry.get("soil_moisture", 0) <= 70:
            # Run sim: +20% Irrigation
            baseline = telemetry
            sim_result = self.sim_engine.run_simulation(baseline, {"irrigation_percent": 20})
            
            yield_gain_pct = sim_result["delta"]["yield_pct"]
            cost_gain_pct = sim_result["delta"]["cost_pct"]
            
            # ROI Heuristic: If Yield gain > Cost increase * 1.5, it's worth it
            # (Assuming Yield value is roughly proportional to cost for this simplified model)
            if yield_gain_pct > 2.0 and yield_gain_pct > cost_gain_pct:
                 recs.append({
                    "type": "AI Optimization",
                    "action": "Increase Irrigation by 20%",
                    "reason": f"Simulation predicts {yield_gain_pct}% yield boost.",
                    "benefit": "optimized yield-to-cost ratio.",
                    "risk": "Low",
                    "source": "AI-What-If"
                })

        # Scenario B: Fertilizer Optimization
        # Try finding if a fertilizer boost works
        sim_result_fert = self.sim_engine.run_simulation(telemetry, {"fertilizer_percent": 10})
        if sim_result_fert["delta"]["yield_pct"] > 3.0:
             recs.append({
                "type": "AI Optimization",
                "action": "Apply 10% Fertilizer Boost",
                "reason": "AI Forecast indicates high nutrient responsiveness.",
                "benefit": f"Projected {sim_result_fert['delta']['yield_pct']}% more yield.",
                "risk": "Low",
                "source": "AI-What-If"
            })
            
        return recs
