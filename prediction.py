import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import glob

# Image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32
# List the classes (subfolders)
classes = ['off', 'on']
predict_data_dir = 'processed_data/test'  # You can use any folder for prediction

# Flow images for prediction
predict_generator = test_datagen.flow_from_directory(
    predict_data_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    class_mode=None,  # Set to None as we are predicting
    shuffle=False
)

# Get the file names for predictions
file_names = predict_generator.filenames

# Load the trained model
loaded_model = tf.keras.models.load_model('custom_image_classifier.h5')

# Make predictions
predictions = loaded_model.predict(predict_generator)

# Map predicted class indices back to class labels
predicted_classes = [classes[i] for i in tf.argmax(predictions, axis=1)]

# Combine file names with predicted classes
results = list(zip(file_names, predicted_classes))

# Print the results
for result in results:
    print(f"File: {result[0]}, Predicted Class: {result[1]}")
