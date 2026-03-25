import joblib
import pandas as pd
import numpy as np
import json
import os
import logging

# from sklearn.ensemble import RandomForestRegressor
# import joblibng.getLogger(__name__)

logger = logging.getLogger(__name__)

class YieldForecastingService:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, "yield_rf_model.pkl")
        self.columns_path = os.path.join(self.base_dir, "yield_columns.pkl")
        self.model = None
        self.model_columns = None
        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.columns_path):
                self.model = joblib.load(self.model_path)
                self.model_columns = joblib.load(self.columns_path)
                logger.info("Yield RF Model loaded successfully.")
            else:
                logger.error("Yield RF Model not found. Run train_yield_model.py.")
        except Exception as e:
            logger.error(f"Failed to load yield model: {e}")

    def predict_yield(self, features):
        """
        Predict yield with uncertainty intervals.
        features: dict containing 'Crop', 'State', 'Season', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide'
        """
        if not self.model or not self.model_columns:
            return {"error": "Model not loaded"}

        # 1. Prepare DataFrame
        input_data = pd.DataFrame([features])
        input_data = pd.get_dummies(input_data)
        
        # 2. Align Columns (Missing cols = 0)
        input_data = input_data.reindex(columns=self.model_columns, fill_value=0)
        
        # 3. Main Prediction (Mean)
        expected_yield = self.model.predict(input_data)[0]
        
        # 4. Uncertainty Estimation (Prediction Interval)
        # Using Random Forest, we can approximate the prediction interval by looking at the variance
        # of the predictions from individual trees.
        
        # Get predictions from all trees
        tree_predictions = []
        for estimator in self.model.estimators_:
            tree_predictions.append(estimator.predict(input_data)[0])
            
        tree_predictions = np.array(tree_predictions)
        
        # Calculate stats
        mean_pred = np.mean(tree_predictions) # Should roughly match expected_yield
        std_dev = np.std(tree_predictions)
        
        # 95% Confidence Interval approx (Mean +/- 1.96 * SD)
        # However, for RF, the error distribution might not be purely normal, but this is a standard approximation.
        best_case = mean_pred + (1.96 * std_dev)
        worst_case = mean_pred - (1.96 * std_dev)
        worst_case = max(0.0, worst_case) # Yield can't be negative
        
        # Confidence Score
        # Inverse of Coefficient of Variation (CV = SD / Mean)
        # Lower CV means trees agree more -> Higher confidence.
        # CV = 0.1 means 10% variation.
        # Score = max(0, 100 * (1 - CV*2)) # heuristic scaling
        if mean_pred > 0:
            cv = std_dev / mean_pred
            confidence_score = max(0, min(100, 100 * (1 - cv)))
        else:
            confidence_score = 0.0
            
        return {
            "expected_yield": round(expected_yield, 2),
            "best_case_yield": round(best_case, 2),
            "worst_case_yield": round(worst_case, 2),
            "confidence_score": round(confidence_score, 1),
            "unit": "tons/hectare"
        }
