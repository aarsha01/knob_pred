import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import glob


# List the classes (subfolders)
classes = ['off', 'on']

# Create folders for training, validation, and testing
train_folder = 'processed_data/train'
val_folder = 'processed_data/validation'
test_folder = 'processed_data/test'

# Image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Number of training samples and validation samples
nb_train_samples = sum(len(files) for _, _, files in os.walk(train_folder))
nb_validation_samples = sum(len(files) for _, _, files in os.walk(val_folder))
# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescale validation and test set
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Flow training images in batches using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Flow validation images in batches using validation_datagen generator
validation_generator = validation_datagen.flow_from_directory(
    val_folder,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the pre-trained VGG16 model without the top layers
base_model = VGG16(weights='imagenet', include_top=False)

# Add custom layers for your classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(classes), activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=5,  # You can adjust the number of epochs
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)

# Save the trained model
model.save('custom_image_classifier.h5')

# Now, let's make predictions on new images
# Set the path to the folder containing images for prediction
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
