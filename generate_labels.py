# import os

# # Paths to the train and validation folders
# train_dir = r"D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\image_classifier_site\dataset\train\other"
# val_dir = r"D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\image_classifier_site\dataset\val\other"

# # Define class names in the same order as in dataset.yaml
# class_names = ['birds', 'fish', 'mammal', 'plant']

# def create_labels(directory):
#     for class_name in class_names:
#         class_folder = os.path.join(directory, class_name)
#         for image_file in os.listdir(class_folder):
#             if image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
#                 label_file = os.path.splitext(image_file)[0] + '.txt'
#                 label_path = os.path.join(class_folder, label_file)
#                 class_index = class_names.index(class_name)

#                 # Dummy bounding box covering entire image
#                 with open(label_path, 'w') as f:
#                     f.write(f"{class_index} 0.5 0.5 1.0 1.0\n")

# # Create labels for training and validation folders
# create_labels(train_dir)
# create_labels(val_dir)

# print("✅ YOLO-compatible dummy label files created.")


import os

# Paths to the 'other' class folders in train and val
train_other_dir = r"D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\image_classifier_site\dataset\train\other"
val_other_dir = r"D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\image_classifier_site\dataset\val\other"

# Class index for 'other'
class_index = 4  # Assuming it's the fifth class (0-based index)

def create_other_labels(directory):
    for image_file in os.listdir(directory):
        if image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(directory, label_file)

            # Dummy bounding box covering entire image
            with open(label_path, 'w') as f:
                f.write(f"{class_index} 0.5 0.5 1.0 1.0\n")

# Create labels for 'other' class only
create_other_labels(train_other_dir)
create_other_labels(val_other_dir)

print("✅ Dummy labels created for 'other' class only.")
