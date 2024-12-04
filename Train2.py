import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Define the paths to the dataset
dataset_dir = 'D:\\Android development\\archive (4)\\train'

# Image dimensions
img_height, img_width = 128, 128


# Function to load images and extract labels from filenames
def load_data(dataset_dir):
    images = []
    hand_labels = []
    finger_labels = []

    for filename in os.listdir(dataset_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(dataset_dir, filename)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (img_width, img_height))
            images.append(image)

            # Extract labels from filename
            label = filename.split('_')[-1].split('.')[0]
            finger_count = int(label[0])  # Number of fingers
            hand_type = 0 if label[1] == 'L' else 1  # 0 for left, 1 for right

            hand_labels.append(hand_type)
            finger_labels.append(finger_count)

    images = np.array(images, dtype=np.float32)
    hand_labels = np.array(hand_labels)
    finger_labels = np.array(finger_labels)

    return images, hand_labels, finger_labels


# Load data
images, hand_labels, finger_labels = load_data(dataset_dir)

# Normalize images
images = images / 255.0

# Convert labels to one-hot encoding
hand_labels = tf.keras.utils.to_categorical(hand_labels, num_classes=2)
finger_labels = tf.keras.utils.to_categorical(finger_labels, num_classes=6)

# Split data into training and test sets
X_train, X_test, hand_y_train, hand_y_test, finger_y_train, finger_y_test = train_test_split(
    images, hand_labels, finger_labels, test_size=0.2, random_state=42)

# Build the model
input_layer = Input(shape=(img_height, img_width, 3))

# Common CNN layers
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layer for hand type
hand_output = Dense(2, activation='softmax', name='hand_type')(x)

# Output layer for finger count
finger_output = Dense(6, activation='softmax', name='finger_count')(x)

# Combine the inputs and outputs into a model
model = Model(inputs=input_layer, outputs=[hand_output, finger_output])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Define a generator function to reduce memory usage during training
def data_generator(X, y_hand, y_finger, batch_size):
    num_samples = len(X)
    while True:
        for offset in range(0, num_samples, batch_size):
            end = offset + batch_size
            batch_X = X[offset:end]
            batch_y_hand = y_hand[offset:end]
            batch_y_finger = y_finger[offset:end]
            yield batch_X, [batch_y_hand, batch_y_finger]


# Set batch size
batch_size = 32


# Create a tf.data.Dataset from the generator function
def generator_wrapper(X, y_hand, y_finger):
    gen = data_generator(X, y_hand, y_finger, batch_size)
    for X_batch, y_batch in gen:
        yield X_batch, (y_batch[0], y_batch[1])


# Create Dataset objects
train_dataset = tf.data.Dataset.from_generator(
    generator_wrapper,
    output_signature=(
        tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 6), dtype=tf.float32)
        )
    ),
    args=(X_train, hand_y_train, finger_y_train)
)

val_dataset = tf.data.Dataset.from_generator(
    generator_wrapper,
    output_signature=(
        tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 6), dtype=tf.float32)
        )
    ),
    args=(X_test, hand_y_test, finger_y_test)
)

# Calculate steps per epoch
train_steps_per_epoch = len(X_train) // batch_size
validation_steps = len(X_test) // batch_size

# Train the model using the generator
model.fit(
    train_dataset,
    steps_per_epoch=train_steps_per_epoch,
    epochs=20,
    validation_data=val_dataset,
    validation_steps=validation_steps
)

# Save the model
model.save('hand_finger_model.h5')

# Evaluate the model
loss, hand_loss, finger_loss, hand_accuracy, finger_accuracy = model.evaluate(X_test, [hand_y_test, finger_y_test])
print(f'Test hand type accuracy: {hand_accuracy * 100:.2f}%')
print(f'Test finger count accuracy: {finger_accuracy * 100:.2f}%')
