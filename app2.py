from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('hand_finger_model.h5')

# Parameters
img_width, img_height = 128, 128
num_classes = 12

# Function to preprocess the frame
def preprocess_frame(frame):
    img = cv2.resize(frame, (img_width, img_height))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to get prediction from the model
def get_prediction(frame):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    predicted_class = np.argmax(prediction)
    fingers = predicted_class // 2
    hand = 'Left' if predicted_class % 2 == 0 else 'Right'
    return fingers, hand

# Video streaming generator function
def generate_frames():
    camera = cv2.VideoCapture(0)  # Use the system camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            fingers, hand = get_prediction(frame)
            text = f'Fingers: {fingers}, Hand: {hand}'
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
