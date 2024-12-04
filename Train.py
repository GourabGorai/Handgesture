import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define the path to the dataset
dataset_path = 'D:\Android development\\archive (3)\leapGestRecog'

# Define the gestures and subjects
gestures = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']
subjects = [f'{i:02d}' for i in range(10)]

# Initialize data and labels lists
data = []
labels = []

# Load and preprocess the images
for subject in subjects:
    subject_path = os.path.join(dataset_path, subject)
    for i, gesture in enumerate(gestures, 1):
        gesture_path = os.path.join(subject_path, f'{i:02d}_{gesture}')
        if os.path.exists(gesture_path):
            for img_name in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    image = cv2.resize(image, (64, 64))  # Resize the images to 64x64 pixels
                    data.append(image)
                    labels.append(i-1)  # Label for the gesture

# Convert data to numpy arrays and normalize
data = np.array(data, dtype='float32') / 255.0
data = np.expand_dims(data, axis=-1)  # Add channel dimension
labels = to_categorical(labels, num_classes=len(gestures))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(gestures), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('hand_gesture_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')
