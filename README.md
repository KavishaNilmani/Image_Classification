**Demo Document: Intelligent Species Classifier Website**

---

### 🌍 Project Overview

The *Intelligent Species Classifier* is a Streamlit-based web application that classifies uploaded images into one of four major biological classes: **birds**, **fish**, **mammals**, and **plants**. It uses a combination of:

* A custom-trained **YOLOv8s classifier** for main class detection
* A **MobileNetV2 model** for identifying specific species in general cases
* The **Hugging Face bird-species-classifier** for fine-grained bird classification
* The **Pl\@ntNet API** for identifying plant species
* **Wikipedia and iNaturalist APIs** for scientific names and descriptive content

---

### 🧪 Tech Stack

* **Frontend**: Streamlit
* **Image Classification Models**:

  * YOLOv8s (custom-trained for 5 classes)
  * MobileNetV2 (pretrained on ImageNet)
  * Hugging Face model: `chriamue/bird-species-classifier`
* **External APIs**:

  * Pl\@ntNet
  * iNaturalist
  * Wikipedia
* **Others**: Torch, TensorFlow, OpenCV, Pillow, Transformers

---

### 🚀 Installation & Setup

#### 1. Clone the Repository

```bash
# Navigate to your workspace
cd image_classifier_site
```

#### 2. Install Dependencies

```bash
pip install -r r.txt
```

#### 3. Add Pl\@ntNet API Key

Create a `.env` file with the following content:

```
PLANTNET_API_KEY=your_api_key_here
```

#### 4. Run the App

```bash
streamlit run app.py
```

---

### 🔧 How to Use

1. Launch the app.
2. Upload an image in JPG/PNG format.
3. The app will:

   * Classify the image into one of: **bird**, **fish**, **mammal**, **plant**, or **other**
   * If **bird**: identify species using Hugging Face model
   * If **plant**: identify species using Pl\@ntNet and fetch from Wikipedia
   * If **mammal/fish**: identify specific object using MobileNetV2 and iNaturalist mapping
4. View class name, confidence score, description, and images (if available).

---

### 📊 Model Training Summary

**Model**: YOLOv8s-cls
**Classes**: 5 (bird, fish, mammal, plant, other)
**Training Images**: 6638
**Validation Images**: 1275
**Epochs**: 10
**Top-1 Accuracy**: 87.6%
**Top-5 Accuracy**: 100%
**Model Size**: \~10.3 MB
**Loss Final Epoch**: 0.0842

```
Fitness Score: 0.9384
Speed: ~13.3ms per inference (CPU)
```

---

### ❌ Known Limitations

* No voice interaction support (can be added in future)
* Accuracy may reduce with noisy/blurred images
* Misclassifications in cases where the object is ambiguous or occluded

---

### ✨ Future Enhancements

* Add voice-based upload and prediction
* Display confidence graphs for all class probabilities
* Provide educational content about each species

---

### 👤 Developed By

**Kavisha Nilmani**

---

