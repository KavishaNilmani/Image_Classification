from ultralytics import YOLO
import sys
from PIL import Image

# Load trained model
model = YOLO("runs/classify/train/weights/best.pt")

# Image path
img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

# Predict
results = model(img_path)
pred = results[0]
cls_id = int(pred.probs.top1)
label = pred.names[cls_id]
conf = pred.probs.data[cls_id].item()

print(f"✅ Predicted: {label} ({conf:.2f})")
