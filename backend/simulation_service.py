import logging
import math

logger = logging.getLogger(__name__)

class AgriSimulationEngine:
    def __init__(self):
        pass

    def run_simulation(self, baseline_data, adjustments):
        """
        Run a rule-based simulation to predict yield and stress under hypothetical conditions.
        
        Args:
            baseline_data (dict): Current telemetry (soil moisture, rainfall, etc.)
            adjustments (dict): User modifications (e.g., {'irrigation_percent': 10, 'temp_change': 2})
            
        Returns:
            dict: Simulated metrics and comparison.
        """
        # 1. Extract Baseline
        # Defaulting to nominal values if missing
        b_soil_moisture = float(baseline_data.get("soil_moisture") or 30.0)
        b_rainfall = float(baseline_data.get("rainfall") or 0.0)
        b_temp = float(baseline_data.get("temperature") or 25.0)
        b_humidity = float(baseline_data.get("humidity") or 60.0)
        
        # 2. Extract Adjustments
        # irrigation_percent: +/- percentage change in water supply
        # temp_change: absolute change in degrees Celsius
        # fertilizer_percent: +/- percentage change in fertilizer
        adj_irrigation_pct = float(adjustments.get("irrigation_percent", 0))
        adj_temp_delta = float(adjustments.get("temp_change", 0))
        adj_fertilizer_pct = float(adjustments.get("fertilizer_percent", 0))
        
        # 3. Apply Adjustments to Environmental Factors
        s_temp = b_temp + adj_temp_delta
        
        # Simulate water input: Rainfall + Irrigation
        # Assumption: Baseline irrigation is "sufficient" (value=1.0 normalized) plus adjustments.
        # Soil moisture response modeled as proportional to input changes.
        # If irrigation increases by 10%, effective water availability increases.
        
        # Model: Effective Water = (Rainfall * 0.8) + (BaseIrrigation * (1 + adj/100))
        # We don't have exact base irrigation volume, so we treat 'soil_moisture' as the proxy for current status.
        # New Moisture = Old Moisture * (1 + combined_water_change) - EvapImpact
        
        # Evaporation estimate (simplified Penman-Monteith proxy)
        # Higher temp = higher evaporation = lower moisture
        evap_factor = 1.0 + (adj_temp_delta * 0.05) # 5% more evap per degree rise
        
        # Water Input Factor
        water_input_factor = 1.0 + (adj_irrigation_pct / 100.0)
        
        s_soil_moisture = (b_soil_moisture * water_input_factor) / evap_factor
        # Cap moisture at 100% and min 0%
        s_soil_moisture = max(0.0, min(100.0, s_soil_moisture))
        
        # 4. Calculate Stress Indices (0.0 to 1.0, where 1.0 is max stress)
        
        # Water Stress
        if s_soil_moisture < 30:
            water_stress = (30 - s_soil_moisture) / 30.0 # High stress if dry
        elif s_soil_moisture > 90:
            water_stress = (s_soil_moisture - 90) / 10.0 # Mild stress if waterlogged
        else:
            water_stress = 0.0
            
        # Heat Stress
        optimal_temp = 25.0
        if s_temp > 35:
            heat_stress = (s_temp - 35) / 10.0
        elif s_temp < 10:
            heat_stress = (10 - s_temp) / 10.0
        else:
            heat_stress = 0.0
        heat_stress = min(1.0, heat_stress)
            
        # Combined Stress
        total_stress = min(1.0, water_stress + heat_stress)
        
        # 5. Yield Projection
        # Baseline Yield assumed to be "Potential Yield" minus current stress
        # Let's assume a base potential of 1000 kg/acre
        base_potential_yield = 1000.0
        
        # Fertilizer impact: Positive adjustment boosts potential, negative reduces it.
        # But diminishing returns apply.
        fert_factor = 1.0
        if adj_fertilizer_pct > 0:
            # log growth for returns: 100% more fert != 100% more yield
            fert_factor = 1.0 + (math.log(1 + (adj_fertilizer_pct/100.0)) * 0.3)
        else:
             # Linear drop for reduction
            fert_factor = 1.0 + (adj_fertilizer_pct / 100.0)
            
        # Stress Reduction Factor (Yield = Potential * (1 - k*Stress))
        # k is crop coefficient, assume 0.8
        yield_reduction_factor = 1.0 - (0.8 * total_stress)
        yield_reduction_factor = max(0.0, yield_reduction_factor)
        
        s_yield = base_potential_yield * fert_factor * yield_reduction_factor
        
        # 6. Cost Calculation
        # Base cost units
        base_cost_irrigation = 500 # currency units
        base_cost_fert = 1000
        
        s_cost_irrigation = base_cost_irrigation * (1 + (adj_irrigation_pct / 100.0))
        s_cost_fert = base_cost_fert * (1 + (adj_fertilizer_pct / 100.0))
        total_cost = s_cost_irrigation + s_cost_fert
        
        # Baseline Cost (for comparison)
        baseline_cost = base_cost_irrigation + base_cost_fert
        
        # 7. Format Result
        return {
            "input": {
                "temp": round(s_temp, 1),
                "moisture": round(s_soil_moisture, 1)
            },
            "metrics": {
                "yield_kg": round(s_yield, 1),
                "cost": round(total_cost, 1),
                "stress_index": round(total_stress * 100, 1) # percentage
            },
            "delta": {
                "yield_pct": round(((s_yield - (base_potential_yield * (1 - 0.8 * 0))) / (base_potential_yield * 1)) * 100, 1), # Approx comp against ideal
                # Better comparison: vs Baseline with 0 adjustments
                "cost_pct": round(((total_cost - baseline_cost) / baseline_cost) * 100, 1)
            }
        }
