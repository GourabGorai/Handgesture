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

# Define the path
dataset_dir = 'D:\\Android development\\archive (4)\\train'

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

# Load images and labels
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

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Check if images and labels are loaded correctly
print(f"Loaded {len(images)} images and {len(labels)} labels.")

if len(images) == 0 or len(labels) == 0:
    raise ValueError("No images or labels loaded. Please check the dataset directory and file naming conventions.")

# Normalize images
images = images / 255.0

# Convert labels to categorical
labels = to_categorical(labels, num_classes=num_classes)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

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
    validation_data=(X_test, y_test)
)

# Save the model
model.save('hand_finger_model.h5')
