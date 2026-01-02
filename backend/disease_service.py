import os
import requests
import numpy as np
import logging

# Setup Logging
logger = logging.getLogger(__name__)

# Placeholder for External API Key
PLANT_ID_API_KEY = "YOUR_PLANT_ID_API_KEY"
PLANT_ID_URL = "https://api.plant.id/v2/identify"

class HybridDiseaseDetector:
    def __init__(self, model_path="plant_disease_model.h5"):
        self.model_path = model_path
        self.local_model = None
        # self.load_local_model() # Uncomment when you have the file

    def load_local_model(self):
        try:
            import tensorflow as tf
            self.local_model = tf.keras.models.load_model(self.model_path)
            logger.info("✅ Local Plant Disease Model Loaded.")
        except Exception as e:
            logger.error(f"⚠️ Failed to load local model: {e}")

    def predict_local(self, image_bytes):
        """
        Run inference using the local CNN.
        Returns: (class_name, confidence)
        """
        if not self.local_model:
            return "Local Model Unavailable", 0.0

        # Preprocessing (Resize, Normalize) - Simplified example
        # import cv2
        # img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        # img = cv2.resize(img, (224, 224))
        # img = img / 255.0
        # predictions = self.local_model.predict(np.expand_dims(img, axis=0))
        # return class_names[np.argmax(predictions)], float(np.max(predictions))
        
        # MOCK RETURN for prototype
        return "Early Blight", 0.85

    def predict_remote(self, image_base64):
        """
        Send image to Plant.id API for expert verification.
        """
        if not PLANT_ID_API_KEY or "YOUR_PLANT_ID" in PLANT_ID_API_KEY:
            logger.warning("⚠️ No External API Key provided.")
            return {"error": "API Key Missing", "suggestions": []}

        headers = {
            "Content-Type": "application/json",
            "Api-Key": PLANT_ID_API_KEY,
        }
        data = {
            "images": [image_base64],
            "modifiers": ["crops_fast", "similar_images"],
            "plant_details": ["common_names", "url", "wiki_description", "taxonomy"]
        }

        try:
            response = requests.post(PLANT_ID_URL, json=data, headers=headers)
            if response.status_code == 200:
                result = response.json()
                # Extract top suggestion
                if result["suggestions"]:
                    top = result["suggestions"][0]
                    return {
                        "disease": top["plant_name"],
                        "confidence": top["probability"],
                        "details": top["plant_details"]
                    }
            return {"error": "API Failed", "suggestions": []}
        except Exception as e:
            logger.error(f"Remote API Error: {e}")
            return {"error": str(e)}

    def analyze(self, image_bytes, image_base64):
        """
        Hybrid Analysis Logic
        """
        # 1. Local Prediction
        local_pred, local_conf = self.predict_local(image_bytes)

        # 2. Remote Verification (Optional / Parallel)
        remote_res = self.predict_remote(image_base64)
        remote_pred = remote_res.get("disease", "Unknown")
        remote_conf = remote_res.get("confidence", 0.0)

        # 3. Consensus Logic
        status = "Uncertain"
        advisory = "Please consult an agronomist."
        
        # Example Advisory Logic
        if local_pred == remote_pred:
            status = "Confirmed"
            advisory = f"Both systems detected {local_pred}. High certainty."
        elif local_conf > 0.9:
            status = "Local Model High Confidence"
            advisory = f"Local model is confident it is {local_pred}."
        
        return {
            "local": {"class": local_pred, "confidence": local_conf},
            "remote": {"class": remote_pred, "confidence": remote_conf},
            "analysis": {"status": status, "advisory": advisory}
        }
