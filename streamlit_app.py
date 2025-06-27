import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from dotenv import load_dotenv
import torch
import io
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
import ssl

# Fix SSL issues
ssl._create_default_https_context = ssl._create_unverified_context

# Load .env for Pl@ntNet key
load_dotenv()
PLANTNET_API_KEY = os.getenv("PLANTNET_API_KEY")

# iNaturalist helper functions
def get_scientific_name_from_common(common_name):
    url = f"https://api.inaturalist.org/v1/search?q={common_name}&sources=taxa"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        for result in data.get("results", []):
            record = result.get("record", {})
            if record.get("rank") == "species":
                return record.get("name")
        return None
    except Exception as e:
        print(f"Error fetching scientific name: {e}")
        return None

# Cache Hugging Face models
CACHE_DIR = "./models/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class_model = YOLO("D:/OneDrive - Lowcode Minds Technology Pvt Ltd/Desktop/image_classifier_site/runs/classify/yolov8s-classifier9/weights/best.pt")
identifier_model = MobileNetV2(weights="imagenet")

@st.cache_resource
def load_bird_model():
    processor = AutoImageProcessor.from_pretrained("chriamue/bird-species-classifier", cache_dir=CACHE_DIR)
    model = AutoModelForImageClassification.from_pretrained("chriamue/bird-species-classifier", cache_dir=CACHE_DIR)
    return processor, model

bird_processor, bird_model = load_bird_model()
main_classes = ["birds", "fish", "mammal", "plant"]

def classify_bird(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = bird_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = bird_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits[0], dim=-1)
    labels = bird_model.config.id2label
    results = [{"label": labels[i], "score": float(p)} for i, p in enumerate(probs)]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]

def identify_plant(image_bytes):
    if not PLANTNET_API_KEY:
        raise ValueError("Pl@ntNet API key not found.")
    files = {
        "organs": (None, "auto"),
        "images": ("image.jpg", image_bytes, "image/jpeg"),
    }
    url = f"https://my-api.plantnet.org/v2/identify/all?include-related-images=true&no-reject=false&nb-results=5&lang=en&api-key={PLANTNET_API_KEY}"
    response = requests.post(url, files=files, verify=False)
    response.raise_for_status()
    return response.json()

st.set_page_config(page_title="🧠 Intelligent Species Classifier", layout="centered")
st.title("🧠 Intelligent Species Identifier")
st.write("Upload an image to classify its type and get detailed information!")

uploaded_file = st.file_uploader("📄 Upload an image", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.stop()

img = Image.open(uploaded_file).convert("RGB")
st.image(img, caption="📷 Uploaded Image", use_column_width=True)

with st.spinner("🔍 Classifying..."):
    result = class_model.predict(img, verbose=False)[0]
    top_index = result.probs.top1
    class_name = result.names[top_index]
    confidence = result.probs.data[top_index].item()

st.markdown(f"### 🏷️ Predicted Class: **{class_name}** (Confidence: {confidence:.2f})")

if class_name.lower() not in main_classes:
    st.warning("⚠️ No description available for this image category.")
    st.stop()

if class_name.lower() == "plant":
    st.write("🌿 Identifying plant species using Pl@ntNet...")
    try:
        image_bytes = uploaded_file.getvalue()
        plant_data = identify_plant(image_bytes)
        if "results" in plant_data and plant_data["results"]:
            st.markdown("### 🌐 Top Plant Matches")
            top_result = plant_data["results"][0]
            top_species = top_result["species"]["scientificNameWithoutAuthor"]
            for r in plant_data["results"][:3]:
                species = r["species"]["scientificNameWithoutAuthor"]
                common_names = r["species"].get("commonNames", [])
                score = r["score"] * 100
                st.markdown(f"**🔎 Species:** {species}")
                if common_names:
                    st.markdown(f"**📝 Common Names:** {', '.join(common_names)}")
                st.markdown(f"**📊 Confidence:** {score:.2f}%")
                st.markdown("---")
            wiki_title = top_species.replace(" ", "_")
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_title}"
            wiki_resp = requests.get(wiki_url, verify=False)
            st.markdown("### 📚 Wikipedia Description")
            if wiki_resp.status_code == 200:
                wd = wiki_resp.json()
                st.markdown(f"**Title:** {wd.get('title')}")
                st.markdown(wd.get("extract", "No description available."))
                if "thumbnail" in wd:
                    st.image(wd["thumbnail"]["source"], width=300)
            else:
                st.warning("ℹ️ Wikipedia has no info on this plant.")
        else:
            st.warning("⚠️ No plant match found.")
    except Exception as e:
        st.error(f"❌ Pl@ntNet error: {e}")

elif class_name.lower() == "birds":
    st.write("🕊️ Identifying bird species using Hugging Face model...")
    try:
        image_bytes = uploaded_file.getvalue()
        results = classify_bird(image_bytes)
        if results:
            st.markdown("### 🐦 Top Bird Predictions")
            for r in results:
                st.markdown(f"**{r['label']}** — {r['score']*100:.2f}%")
            top_bird = results[0]["label"]
            sci_name = get_scientific_name_from_common(top_bird)
            wiki_title = sci_name.replace(" ", "_") if sci_name else top_bird.replace(" ", "_")
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_title}"
            wiki_resp = requests.get(wiki_url, verify=False)
            st.markdown(f"### 📚 Wikipedia Description for {top_bird}")
            if wiki_resp.status_code == 200:
                wd = wiki_resp.json()
                st.markdown(f"**Title:** {wd.get('title')}")
                st.markdown(wd.get("extract", "No description available."))
                if "thumbnail" in wd:
                    st.image(wd["thumbnail"]["source"], width=300)
            else:
                st.warning("ℹ️ Wikipedia has no info on this bird.")
        else:
            st.warning("⚠️ No bird species identified.")
    except Exception as e:
        st.error(f"❌ Bird classifier error: {e}")

else:
    st.write("🔎 Identifying object using MobileNetV2...")
    try:
        resized = img.resize((224, 224))
        arr = keras_image.img_to_array(resized)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        preds = identifier_model.predict(arr)
        top_label = decode_predictions(preds, top=1)[0][0][1]

        # Mislabel fix via iNaturalist
        suggested = get_scientific_name_from_common(top_label)
        override = st.text_input(
            "🤔 Don't agree? Enter correct object name:",
            value=suggested if suggested else top_label.replace("_", " ")
        )

        label = override.strip().replace(" ", "_")
        st.success(f"✅ Identified as: **{override.title()}**")

        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{label}"
        wiki_resp = requests.get(wiki_url, verify=False)
        if wiki_resp.status_code == 200:
            wd = wiki_resp.json()
            st.markdown("### 🌐 Description")
            st.markdown(f"**Title:** {wd.get('title')}")
            st.markdown(wd.get("extract", "No description available."))
            if "thumbnail" in wd:
                st.image(wd["thumbnail"]["source"], width=300)
        else:
            st.warning("ℹ️ Wikipedia has no info on this item.")
    except Exception as e:
        st.error(f"❌ Error identifying specific object: {e}")
