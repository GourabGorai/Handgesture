from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

app = Flask(__name__)

# Load the trained models
gesture_model = tf.keras.models.load_model('hand_gesture_model.h5')
finger_model = load_model('hand_finger_model.h5')

# Define the gestures
gestures = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']

# Parameters for hand finger count model
img_width, img_height = 128, 128

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def preprocess_gesture_image(image):
    """ Preprocess the image for gesture prediction. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=(0, -1))
    return reshaped

def preprocess_finger_image(image):
    """ Preprocess the image for finger count prediction. """
    img = cv2.resize(image, (img_width, img_height))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_gesture(image):
    """ Predict the hand gesture. """
    preprocessed = preprocess_gesture_image(image)
    prediction = gesture_model.predict(preprocessed)
    gesture = gestures[np.argmax(prediction)]
    return gesture

def predict_fingers(image):
    """ Predict the number of fingers and hand side. """
    processed_frame = preprocess_finger_image(image)
    prediction = finger_model.predict(processed_frame)
    predicted_class = np.argmax(prediction)
    fingers = predicted_class // 2
    hand = 'Left' if predicted_class % 2 == 0 else 'Right'
    return fingers, hand

def detect_hand(frame):
    """ Detect the hand in the frame using MediaPipe. """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        return True, frame
    return False, frame

def generate_frames(feature):
    """ Capture video frames from the camera and predict based on selected feature. """
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            hand_detected, frame = detect_hand(frame)
            if hand_detected:
                if feature == 'gesture':
                    gesture = predict_gesture(frame)
                    cv2.putText(frame, f'Gesture: {gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                elif feature == 'fingers':
                    fingers, hand = predict_fingers(frame)
                    cv2.putText(frame, f'Fingers: {fingers}, Hand: {hand}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'No hand detected', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """ Render the home page with feature selection. """
    return render_template('index.html')

@app.route('/select_feature', methods=['POST'])
def select_feature():
    """ Handle feature selection and redirect to the video feed. """
    feature = request.form.get('feature')
    return render_template('video_feed.html', feature=feature)

@app.route('/video_feed/<feature>')
def video_feed(feature):
    """ Video streaming route. """
    return Response(generate_frames(feature), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
