from ultralytics import YOLO

# Path to the dataset YAML file
data_yaml = r'D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\image_classifier_site\dataset\dataset.yaml'

# Initialize the YOLOv8 model for classification
model = YOLO('model/yolov8s-cls.pt')  # Load a pretrained YOLOv8 classification model
# model = YOLO('yolov5s-cls.pt')

# Train the model with the data configuration (YAML file)
model.train(
    data=data_yaml,  # Use the dataset YAML file
    epochs=20,
    imgsz=224,
    batch=16
)

# Save the trained model manually
model.export(format='pt')  # Save the model as best.pt

# Print completion message
print("Training complete and model saved!")
