import tensorflow as tf
import os

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "plant_disease_model.h5")

def download_and_save_model():
    print("Downloading MobileNetV2 (ImageNet weights)...")
    
    # Load MobileNetV2 base model (pre-trained on ImageNet)
    # We include the top because we want a ready-to-use classifier for testing, 
    # OR we can exclude top and add our own small head for the 3 classes we have in the code.
    # The current code expects 3 classes: "Healthy", "Early Blight", "Late Blight".
    # ImageNet has 1000 classes.
    # So we MUST add a custom head.
    
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Create new model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax') # 3 classes
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Saving model...")
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Model size: {os.path.getsize(MODEL_SAVE_PATH) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    download_and_save_model()
