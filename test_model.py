import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('model/efficientnetb0_model.h5')

# Define your known classes (must match training folder order)
class_labels = ['birds', 'fish', 'mammal', 'plant']

# Load and preprocess the image
img_path = 'test_images/leave.jpg'  # 🖼️ use any image not in the 4 classes
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)

# Get top prediction and confidence
confidence = np.max(predictions[0])
predicted_index = np.argmax(predictions[0])
predicted_label = class_labels[predicted_index]

# Set threshold (e.g., 85%)
THRESHOLD = 0.85

# Decision logic
if confidence < THRESHOLD:
    print("🛑 Not Found. Not in Supported Categories.")
    # print(f"🔍 Top prediction: {predicted_label} ({confidence*100:.2f}%) but confidence is too low.")
else:
    print(f"✅ Predicted class: {predicted_label}")
    print(f"📊 Confidence: {confidence*100:.2f}%")
