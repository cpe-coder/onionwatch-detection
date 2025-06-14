import cv2
import numpy as np
import tensorflow as tf
import os
from datetime import datetime

# === CONFIGURATION ===
CAM_URL = "http://192.168.43.249/cam-hi.jpg"  # Change to your ESP32-CAM snapshot URL
SAVE_DIR = "captures"
CONFIDENCE_THRESHOLD = 0.7
IMAGE_SIZE = 256
DIFF_THRESHOLD = 25  # Sensitivity for movement detection

# === Prepare environment ===
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load model ===
model = tf.keras.models.load_model("./model/onionwatch.keras")  # or .h5
class_names = ['Bangag', 'BlackBug', 'Bugs']  # Your actual classes

# === Preprocessing ===
def preprocess_image(img):
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_bug(img):
    processed = preprocess_image(img)
    predictions = model.predict(processed)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    if confidence >= CONFIDENCE_THRESHOLD:
        return class_names[class_idx], confidence
    return None, None

# === Frame difference setup ===
prev_frame = None

cv2.namedWindow("ESP32-CAM Stream", cv2.WINDOW_NORMAL)
print("📡 Starting OnionWatch AI (Press 'q' to quit)")
while True:
    try:
        cap = cv2.VideoCapture(CAM_URL)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print("⚠️ Failed to read frame")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray_blur
            cv2.imshow("ESP32-CAM Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # === Compute absolute difference between frames ===
        frame_diff = cv2.absdiff(prev_frame, gray_blur)
        thresh = cv2.threshold(frame_diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bug_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Adjust area threshold if needed
                bug_detected = True
                break

        # === If bug/insect is detected ===
        if bug_detected:
            print("🪲 Movement detected! Predicting...")
            predicted_class, confidence = predict_bug(frame)
            if predicted_class:
                filename = f"{SAVE_DIR}/{predicted_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✅ Detected {predicted_class} ({confidence*100:.2f}%). Saved at {filename}")
            else:
                print("❌ Movement detected, but no insect confidently classified.")

        # Show the stream
        cv2.imshow("ESP32-CAM Stream", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update the previous frame
        prev_frame = gray_blur

    except Exception as e:
        print(f"🚨 Error: {e}")
        continue

cv2.destroyAllWindows()
