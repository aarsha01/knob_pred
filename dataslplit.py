import os
import shutil
from sklearn.model_selection import train_test_split

# Set the path to the 'data' folder
data_folder = 'data'

# List the classes (subfolders)
classes = ['off', 'on']

# Create folders for training, validation, and testing
train_folder = 'processed_data/train'
val_folder = 'processed_data/validation'
test_folder = 'processed_data/test'

# Create subfolders for each class in train, validation, and test folders
for class_name in classes:
    os.makedirs(os.path.join(train_folder, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_folder, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_folder, class_name), exist_ok=True)

# Split data into training, validation, and test sets
for class_name in classes:
    class_folder = os.path.join(data_folder, class_name)
    images = os.listdir(class_folder)
    train_images, test_val_images = train_test_split(images, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(test_val_images, test_size=0.5, random_state=42)

    # Copy images to respective folders
    for image in train_images:
        shutil.copy(os.path.join(class_folder, image), os.path.join(train_folder, class_name, image))
    
    for image in val_images:
        shutil.copy(os.path.join(class_folder, image), os.path.join(val_folder, class_name, image))
    
    for image in test_images:
        shutil.copy(os.path.join(class_folder, image), os.path.join(test_folder, class_name, image))

print("Data split successfully.")
