import streamlit as st
import uuid
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
import requests

# --- CONFIG ---
st.set_page_config(page_title="🧠 Smart Image Classifier", layout="centered")
st.title("🖼️ Smart Image Classifier (Class + Wikipedia Info)")

# --- LOAD MODELS ---
custom_model = load_model('model/mobilenet_model.h5')  # Your 4-class model
imagenet_model = MobileNetV2(weights="imagenet")        # Pretrained ImageNet model
class_names = ['mammal', 'bird', 'fish', 'plant']

# --- PREDICT GENERAL CLASS ---
def predict_general_class(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = custom_model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index]
    return predicted_class, confidence

# --- PREDICT SPECIFIC LABEL (IMAGENET) ---
def predict_specific_object(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))
    preds = imagenet_model.predict(img_array)
    label = decode_predictions(preds, top=1)[0][0][1].replace('_', ' ')
    confidence = decode_predictions(preds, top=1)[0][0][2]
    return label, confidence

# --- WIKIPEDIA FETCH ---
def get_wikipedia_info(label):
    try:
        label = label.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{label}"
        res = requests.get(url)
        if res.status_code == 200:
            return res.json().get("extract", "No description found.")
        else:
            return "No description found."
    except Exception as e:
        return f"Error fetching Wikipedia data: {e}"

# --- UI ---
temp_dir = Path("temp_images")
temp_dir.mkdir(exist_ok=True)

uploaded_file = st.file_uploader("📤 Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Save image temporarily
    img = Image.open(uploaded_file)
    file_path = temp_dir / f"{uuid.uuid4().hex}.jpg"
    img.save(file_path)

    # Display image
    st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)
    st.write("🔍 Classifying...")

    # Step 1: General class prediction
    general_class, confidence = predict_general_class(file_path)
    st.success(f"✅ General Class: **{general_class.capitalize()}** ({confidence * 100:.2f}%)")

    # Step 2: Specific object prediction
    specific_label, specific_conf = predict_specific_object(file_path)
    st.info(f"🔎 Specific Prediction: **{specific_label.title()}** ({specific_conf * 100:.2f}%)")

    # Step 3: Wikipedia Info
    description = get_wikipedia_info(specific_label)
    st.markdown("📚 **Description:**")
    st.write(description)

    # Clean up
    file_path.unlink()
