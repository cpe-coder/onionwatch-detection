import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = load_model("./model/onionwatch.keras")

class_names = ['Bangag', 'BlackBug', 'Bugs']  

ESP32_CAM_URL = "http://<ESP32-CAM-IP>/stream" 

cap = cv2.VideoCapture(ESP32_CAM_URL)

fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

print("🔍 Starting Insect Detection...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame = cv2.resize(frame, (640, 480))
    fgmask = fgbg.apply(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500: 
            x, y, w, h = cv2.boundingRect(cnt)

            roi = frame[y:y+h, x:x+w]
            try:
                roi_resized = cv2.resize(roi, (224, 224))
                image = img_to_array(roi_resized)
                image = preprocess_input(image)
                image = np.expand_dims(image, axis=0)

                predictions = model.predict(image)
                pred_idx = np.argmax(predictions[0])
                label = class_names[pred_idx]
                confidence = predictions[0][pred_idx]

                text = f"{label} ({confidence*100:.1f}%)"
                color = (0, 255, 0) if label != "background" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if confidence > 0.85 and label != "background":
                    filename = f"{label}_{int(confidence*100)}.jpg"
                    cv2.imwrite(filename, roi)
                    print(f"📸 Saved: {filename}")

            except Exception as e:
                print(f"⚠️ Error processing ROI: {e}")

    cv2.imshow("Onion Farm Insect Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
