import cv2
import numpy as np
import tensorflow as tf
import os
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db

CAM_URL = "http://192.168.43.249:81/stream"
SAVE_DIR = "captures"
CONFIDENCE_THRESHOLD = 1.0
IMAGE_SIZE = 256
CAPTURE_INTERVAL = 5 

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://onionwatch-a9a56-default-rtdb.firebaseio.com/'  
})
firebase_ref = db.reference("insect-detections")

model = tf.keras.models.load_model("./model/onionwatch.keras")
class_names = ['Bangag', 'BlackBug', 'Bugs']

os.makedirs(SAVE_DIR, exist_ok=True)

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

print("📡 Press SPACE to start/stop capture | Press ENTER to predict/send | Press Q to quit")

cv2.namedWindow("ESP32-CAM Stream", cv2.WINDOW_NORMAL)

capturing = False
last_capture_time = 0
image_count = 0

while True:
    try:
        cap = cv2.VideoCapture(CAM_URL)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print("⚠️ Failed to read frame")
            continue

        now = time.time()

        if capturing and now - last_capture_time >= CAPTURE_INTERVAL:
            filename = f"{SAVE_DIR}/img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            image_count += 1
            last_capture_time = now
            print(f"📸 Captured: {filename}")

        cv2.imshow("ESP32-CAM Stream", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '): 
            capturing = not capturing
            state = "started" if capturing else "stopped"
            print(f"⏯️ Capture {state}.")

        elif key == 13: 
            print("🔍 Predicting saved images...")
            send_count = 0
            for fname in os.listdir(SAVE_DIR):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(SAVE_DIR, fname)
                    img = cv2.imread(path)

                    if img is None:
                        continue

                    predicted_class, confidence = predict_bug(img)
                    if predicted_class:
                        print(f"✅ {fname} = {predicted_class} ({confidence*100:.2f}%)")

                        data = {
                            "filename": fname,
                            "predicted_class": predicted_class,
                            "confidence": float(confidence),
                            "timestamp": datetime.now().isoformat()
                        }
                        firebase_ref.push(data)
                        send_count += 1
                    else:
                        print(f"❌ {fname} not confidently classified.")

                    os.remove(path)

            print(f"📤 Done. {send_count} classified results sent to Firebase.")

        elif key == ord('q'):
            print("👋 Exiting...")
            break

    except Exception as e:
        print(f"🚨 Error: {e}")
        continue

cv2.destroyAllWindows()
