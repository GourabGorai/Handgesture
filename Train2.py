import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Ensure PIL is installed
try:
    from PIL import Image
except ImportError:
    raise ImportError("Please install the Pillow library to use this script: pip install pillow")

# Define the paths for training and testing datasets
train_dataset_dir = 'D:\\Android development\\archive (4)\\train'
test_dataset_dir = 'D:\\Android development\\archive (4)\\test'

# Parameters
img_width, img_height = 128, 128
num_classes = 12  # 0L, 0R, 1L, 1R, ..., 5L, 5R

# Function to extract labels from filenames
def extract_label(filename):
    match = re.match(r'(\d)([LR])\d*\.png', filename)
    if match:
        fingers = int(match.group(1))
        hand = match.group(2)
        label_index = fingers * 2 + (0 if hand == 'L' else 1)
        return label_index
    else:
        raise ValueError(f"Filename {filename} does not match expected pattern")

# Function to load images and labels from a directory
def load_images_labels(dataset_dir):
    images = []
    labels = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(dataset_dir, filename)
            try:
                img = load_img(img_path, target_size=(img_width, img_height))
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(extract_label(filename))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

# Load training images and labels
train_images, train_labels = load_images_labels(train_dataset_dir)

# Load testing images and labels
test_images, test_labels = load_images_labels(test_dataset_dir)

# Check if images and labels are loaded correctly
print(f"Loaded {len(train_images)} training images and {len(train_labels)} training labels.")
print(f"Loaded {len(test_images)} testing images and {len(test_labels)} testing labels.")

if len(train_images) == 0 or len(train_labels) == 0 or len(test_images) == 0 or len(test_labels) == 0:
    raise ValueError("No images or labels loaded. Please check the dataset directories and file naming conventions.")

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert labels to categorical
train_labels = to_categorical(train_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # 12 classes: 0L, 0R, 1L, 1R, ..., 5L, 5R
])

# Compile the model
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=25,
    validation_data=(X_val, y_val)
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model.save('hand_finger_model.h5')