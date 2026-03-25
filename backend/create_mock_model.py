import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
import os

MODEL_SAVE_PATH = "plant_disease_model.h5"

def create_mock_model():
    print("Creating mock model...")
    
    # Simple CNN structure that expects 224x224 RGB images
    model = Sequential([
        tf.keras.Input(shape=(224, 224, 3)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        # Output layer with 3 example classes: "Healthy", "Early Blight", "Late Blight"
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.save(MODEL_SAVE_PATH)
    print(f"Mock model saved to {os.path.abspath(MODEL_SAVE_PATH)}")

if __name__ == "__main__":
    create_mock_model()
