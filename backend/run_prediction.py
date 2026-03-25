import numpy as np
import tensorflow as tf
from disease_service import HybridDiseaseDetector
import base64

# Simple mock image (random noise)
def create_mock_image():
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

def run_test():
    print("Initializing Disease Detector...")
    detector = HybridDiseaseDetector()
    
    # Create valid mock image bytes
    from PIL import Image
    import io
    
    img = Image.fromarray(create_mock_image())
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    print("Running Prediction on Random Noise...")
    result = detector.analyze(image_bytes, image_base64)
    print("Result:", result)
    
    if "local" in result and result["local"]["confidence"] >= 0:
        print("Test PASSED: Prediction was successful.")
    else:
        print("Test FAILED.")

if __name__ == "__main__":
    run_test()
