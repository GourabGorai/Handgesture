from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('hand_gesture_model.h5')

# Define the gestures
gestures = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']

def preprocess_image(image):
    """ Preprocess the image for prediction. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=(0, -1))
    return reshaped

def generate_frames():
    """ Capture video frames from the camera and predict gestures. """
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Preprocess the frame
            preprocessed = preprocess_image(frame)

            # Predict the gesture
            prediction = model.predict(preprocessed)
            gesture = gestures[np.argmax(prediction)]

            # Display the prediction on the frame
            cv2.putText(frame, f'Gesture: {gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame to the browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """ Render the home page. """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """ Video streaming route. """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
