from ultralytics import YOLO
import os
from PIL import Image

# Load YOLO-World model (make sure this file exists locally)
model = YOLO("model/yolov8s-world.pt")

# Set your custom classes (open-vocabulary setup)
custom_classes = ['mammal', 'bird', 'fish', 'plant']
model.set_classes(custom_classes)

# Define dataset path
dataset_dir = 'dataset/training'

# Output folder for annotated images
output_dir = 'predicted_images'
os.makedirs(output_dir, exist_ok=True)

# Process each class folder
for class_name in custom_classes:
    class_folder = os.path.join(dataset_dir, class_name)

    if not os.path.isdir(class_folder):
        print(f"⚠️ Folder not found: {class_folder}")
        continue

    for img_file in os.listdir(class_folder):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(class_folder, img_file)

            # Run inference (no need to pass `classes=` anymore)
            results = model(img_path, conf=0.25)
            result = results[0]

            print(f"🔍 {img_file} → Detected:")

            if result.boxes:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    label = custom_classes[cls_id] if cls_id < len(custom_classes) else "Unknown"
                    print(f"  ✅ {label} ({conf:.2f})")
            else:
                print("  ❌ No object detected")

            # Save the annotated image
            out_path = os.path.join(output_dir, f"pred_{class_name}_{img_file}")
            result.save(filename=out_path)
