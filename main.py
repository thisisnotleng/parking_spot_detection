# === Import Required Libraries ===
# These include Flask for web serving, OpenCV for video processing,
# NumPy and TensorFlow/Keras for ML model usage,
# and custom module 'send_alert' for notifications.
from flask import Flask, render_template, Response, jsonify
import cv2
import pickle
import numpy as np
from keras.models import load_model
import time
import tensorflow as tf
from send_alert import alert_wrong_parking  # Handles Telegram alert sending

# === Initialize Flask App ===
app = Flask(__name__)

# === GPU Configuration for TensorFlow (Improves Performance) ===
# Ensures the app uses available GPU (if any) to accelerate inference.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')  # Use only the first GPU
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available, running on CPU")

# === Load Trained Deep Learning Model ===
# This model classifies cropped parking spot images into 3 classes.
model = load_model('model_final.h5')

# === Map Model Output Indices to Human-Readable Labels ===
class_dictionary = {0: 'no_car', 1: 'car', 2: 'wrong_parking'}

# === Load Saved Parking Spot Coordinates from Pickle File ===
# This file contains positions of each parking spot in the video.
with open('carposition.pkl', 'rb') as f:
    positionList = pickle.load(f)

# === Ensure Position Format is Consistent and Well-Structured ===
# Each entry should be ((x, y), spot_number)
fixed_list = []
for i, item in enumerate(positionList):
    if isinstance(item, tuple):
        if len(item) == 2:
            if isinstance(item[0], tuple) and isinstance(item[1], int):
                fixed_list.append(item)
            elif isinstance(item[0], int) and isinstance(item[1], int):
                fixed_list.append(((item[0], item[1]), i + 1))
            else:
                print(f"❌ Unrecognized format at index {i}: {item}")
        else:
            print(f"❌ Tuple of unexpected length at index {i}: {item}")
    else:
        print(f"❌ Not a tuple at index {i}: {item}")
positionList = fixed_list  # Replace with cleaned version

# === Define Parameters and Tracking Variables ===
width, height = 130, 65  # Size of each parking spot to crop
confirmation_duration = 5  # Seconds to confirm parking status change

# Dictionaries to track state confirmation and prevent flickering
state_start_time = {}  # Time when a state started
current_state = {}     # Currently predicted state
alerted_once = []      # Keeps track of which spots have been alerted

# === Load Video Feed ===
# This video simulates live parking lot input
cap = cv2.VideoCapture('car_test_-car2.mp4')  # You can swap with another test video

# === Function to Detect and Classify Parking Spots in Each Frame ===
def checkParkingSpace(img):
    imgCrops = []         # Store cropped spot images
    spot_info = []        # Store position and number for drawing
    spaceCounter = 0      # Count of free parking spots
    wrongParkingCounter = 0  # Count of wrongly parked cars

    # === Crop and Preprocess Each Spot ===
    for item in positionList:
        pos, spot_number = item
        x, y = pos
        cropped_img = img[y:y+height, x:x+width]
        imgResized = cv2.resize(cropped_img, (48, 48))  # Resize to model input size
        imgNormalized = imgResized / 255.0              # Normalize pixel values
        imgCrops.append(imgNormalized)
        spot_info.append({'position': pos, 'number': spot_number})

    # Convert list to numpy array for batch prediction
    imgCrops = np.array(imgCrops)

    # === Run Predictions on Cropped Spot Images ===
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        predictions = model.predict(imgCrops, verbose=0)

    # === Analyze Each Prediction ===
    for i, spot in enumerate(spot_info):
        pos = spot['position']
        spot_number = spot['number']
        x, y = pos
        intId = np.argmax(predictions[i])  # Get predicted class index
        label = class_dictionary[intId]    # Convert to label (string)

        # === Handle Confirmation Timing ===
        # Prevent flickering by requiring consistent prediction for X seconds
        if pos not in current_state:
            current_state[pos] = label
            state_start_time[pos] = time.time()

        if label == current_state[pos]:
            elapsed = time.time() - state_start_time[pos]
            confirmed = elapsed >= confirmation_duration
        else:
            current_state[pos] = label
            state_start_time[pos] = time.time()
            confirmed = False

        # === Visualization Based on Classification Result ===
        # Choose color, thickness, and handle alerts
        if label == 'no_car':
            color = (0, 255, 0) if confirmed else (150, 255, 150)
            thickness = 5
            textColor = (0, 0, 0)
            if confirmed:
                spaceCounter += 1
                if spot_number in alerted_once:
                    alerted_once.remove(spot_number)  # Reset alert flag
        elif label == 'wrong_parking':
            color = (255, 255, 0) if confirmed else (255, 255, 150)
            thickness = 2
            textColor = (255, 255, 255)
            if confirmed:
                wrongParkingCounter += 1
                if spot_number not in alerted_once:
                    alert_wrong_parking(spot_number)  # Send Telegram alert
                    alerted_once.append(spot_number)
        else:  # Properly parked car
            color = (0, 0, 255) if confirmed else (150, 150, 255)
            thickness = 2
            textColor = (255, 255, 255)
            if spot_number in alerted_once:
                alerted_once.remove(spot_number)

        # === Draw Rectangles and Labels on Frame ===
        cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height), color, thickness)

        # Draw label box
        font_scale = 0.5
        text_thickness = 1
        textSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
        textX = x
        textY = y + height - 5
        cv2.rectangle(img, (textX, textY - textSize[1] - 5), (textX + textSize[0] + 6, textY + 2), color, -1)
        cv2.putText(img, label, (textX + 3, textY - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, textColor, text_thickness)
        
        # Draw spot number
        cv2.putText(img, str(spot_number), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # === Show Status Summary on Frame ===
    totalSpaces = len(positionList)
    occupied_spaces = totalSpaces - spaceCounter - wrongParkingCounter

    cv2.putText(img, f'Space Count: {spaceCounter}', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f'Wrong Parking Count: {wrongParkingCounter}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return img, spaceCounter, wrongParkingCounter, occupied_spaces

# === Generator Function to Stream Processed Video Frames ===
def generate_frames():
    while True:
        # Restart video if it ends (loop playback)
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (1280, 720))  # Resize for consistent display
        img, free_spaces, wrong_parking_spaces, occupied_spaces = checkParkingSpace(img)

        # Encode frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()

        # Yield frame with MJPEG header
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

# === Flask Routes ===

@app.route('/')
def index():
    # Load the homepage with embedded video feed
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Route for live video stream
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/space_count')
def space_count():
    # API route to get current parking statistics (can be polled by frontend)
    success, img = cap.read()
    if success:
        img = cv2.resize(img, (1280, 720))
        _, free_spaces, wrong_parking_spaces, occupied_spaces = checkParkingSpace(img)
        return jsonify(free=free_spaces, wrong=wrong_parking_spaces, occupied=occupied_spaces)
    return jsonify(free=0, wrong=0, occupied=0)

# === Start the Flask Web Server ===
if __name__ == "__main__":
    app.run(debug=True)
