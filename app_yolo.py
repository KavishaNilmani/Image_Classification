import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image

st.set_page_config(page_title="YOLO-World Classifier", layout="centered")

# Load YOLO-World model
model = YOLO("model/yolov8s-world.pt")

# Define and set custom classes
custom_classes = ['mammal', 'bird', 'fish', 'plant']
model.set_classes(custom_classes)  # ✅ Correct way for YOLO-World

st.title("🧠 YOLO-World Image Classifier")
st.markdown("Upload an image and detect whether it's a **mammal, bird, fish, or plant**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save temporarily
    temp_path = "temp.jpg"
    img.save(temp_path)

    # Run model
    with st.spinner("Classifying..."):
        results = model(temp_path, conf=0.25)  # ❌ no `classes=` here!
        result = results[0]

        if result.boxes:
            class_id = int(result.boxes.cls[0])
            conf = result.boxes.conf[0].item()
            label = custom_classes[class_id]
            st.success(f"✅ Detected: **{label}** ({conf:.2f})")
        else:
            st.error("❌ No class detected.")

        # Save and show prediction image
        output_path = os.path.join("predicted_images", f"result_{uploaded_file.name}")
        os.makedirs("predicted_images", exist_ok=True)
        result.save(filename=output_path)
        st.image(output_path, caption="Detected Output", use_column_width=True)

    # Clean up
    os.remove(temp_path)
