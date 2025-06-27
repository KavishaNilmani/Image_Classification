import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(page_title="YOLOv8 Classifier", layout="centered")

# Load model
model = YOLO("runs/classify/train/weights/best.pt")

st.title("🧠 YOLOv8 Image Classifier")
st.markdown("Upload an image to classify it as **bird, fish, mammal, or plant**.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        results = model(img)
        pred = results[0]
        cls_id = int(pred.probs.top1)
        label = pred.names[cls_id]
        conf = pred.probs.data[cls_id].item()
        st.success(f"✅ Predicted: **{label}** ({conf:.2f})")
