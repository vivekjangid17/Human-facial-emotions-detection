# webcam.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "emotion_model.keras"
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

model = load_model(MODEL_PATH)


def preprocess_face(face_img):
    face = cv2.resize(face_img, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)   # (48,48,1)
    face = np.expand_dims(face, axis=0)    # (1,48,48,1)
    return face


def predict_emotion(face_img):
    processed = preprocess_face(face_img)
    prediction = model.predict(processed, verbose=0)[0]
    class_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    return CLASS_NAMES[class_idx], confidence


def detect_emotions_in_frame(frame):
    result_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    predictions = []

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]

        if face_gray.size == 0:
            continue

        label, confidence = predict_emotion(face_gray)
        predictions.append({
            "box": (x, y, w, h),
            "emotion": label,
            "confidence": confidence
        })

        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            result_frame,
            f"{label} ({confidence:.1%})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return result_frame, predictions


def detect_emotions_in_image(image):
    return detect_emotions_in_frame(image)


def start_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame, _ = detect_emotions_in_frame(frame)
        cv2.imshow("Webcam Emotion Detection", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_webcam()