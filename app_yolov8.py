import streamlit as st
from ultralytics import YOLO
import requests
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import ssl

# Fix SSL verification for Wikipedia API
ssl._create_default_https_context = ssl._create_unverified_context

# Load YOLOv8 classification model
class_model = YOLO(r'D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\image_classifier_site\runs\classify\yolov8s-classifier9\weights\best.pt')

# Load MobileNetV2 for specific object recognition
identifier_model = MobileNetV2(weights="imagenet")

# Define main class labels
main_classes = ['birds', 'fish', 'mammal', 'plant']

# Streamlit UI setup
st.set_page_config(page_title="🧠 Intelligent Species Classifier", layout="centered")
st.title("🧠 Intelligent Species Identifier")
st.write("Upload an image to classify its type and get detailed information!")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Uploaded Image", use_column_width=True)

    # Step 1: Classify the image into one of 5 categories using YOLOv8
    with st.spinner("🔍 Classifying..."):
        result = class_model.predict(img, verbose=False)[0]
        top_index = result.probs.top1
        class_name = result.names[top_index]
        confidence = result.probs.data[top_index].item()

    st.markdown(f"### 🏷️ Predicted Class: **{class_name}** (Confidence: {confidence:.2f})")

    # Step 2: If class is valid (not 'Other'), do further analysis
    if class_name.lower() in main_classes:
        st.write("🔎 Identifying specific object...")

        try:
            # Preprocess image for MobileNetV2
            resized_img = img.resize((224, 224))
            img_array = keras_image.img_to_array(resized_img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Predict with MobileNetV2
            preds = identifier_model.predict(img_array)
            # label = decode_predictions(preds, top=1)[0][0][1]
            top_label = decode_predictions(preds, top=1)[0][0][1]

            # st.success(f"✅ Identified as: **{label.replace('_', ' ').title()}**")

            # Optional override
            override_label = st.text_input("🤔 Don't agree? Enter correct object name:", value=top_label.replace("_", " "))

            label = override_label.strip().replace(" ", "_")
            st.success(f"✅ Identified as: **{override_label.title()}**")

            # Wikipedia summary (via API)
            # wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{label.replace(' ', '_')}"
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{label}"
            wiki_response = requests.get(wiki_url, verify=False)

            if wiki_response.status_code == 200:
                wiki_data = wiki_response.json()
                st.markdown("### 🌐 Description")
                st.markdown(f"**Title:** {wiki_data.get('title')}")
                st.markdown(wiki_data.get('extract', 'No description available.'))

                if "thumbnail" in wiki_data:
                    st.image(wiki_data["thumbnail"]["source"], width=300)
            else:
                st.warning("ℹ️ Wikipedia has no info on this item.")

        except Exception as e:
            st.error(f"❌ Error identifying specific object: {e}")
    else:
        # For 'Other' class, skip description
        st.warning("⚠️ No description available for this image category (classified as 'Other').")
