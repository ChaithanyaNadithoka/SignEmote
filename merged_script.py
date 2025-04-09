import numpy as np
import cv2
import tensorflow as tf
import pickle
import mediapipe as mp
import pyttsx3
import threading
import time  # Import time module for delay

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize text-to-speech engine
tts = pyttsx3.init()
tts.setProperty('rate', 150)
tts.setProperty('volume', 1.0)

# Load models
face_model = tf.keras.models.load_model('face_model_new.h5')
sign_model = tf.keras.models.load_model('sign_language_model.h5')

# Load sign language label encoder
with open('label_encoder.p', 'rb') as f:
    label_encoder = pickle.load(f)
sign_classes = label_encoder.classes_

# Define emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Mood mapping
mood_mapping = {
    ("Happy", "Surprised"): "Positive Mood",
    ("Neutral", "Slightly Sad"): "Calm/Relaxed",
    ("Sad", "Angry", "Disgusted", "Fearful"): "Low Mood",
    ("Happy", "Sad"): "Fluctuating Mood"
}

# Function to get mood from emotion
def get_mood(emotion):
    for emotions, mood in mood_mapping.items():
        if emotion in emotions:
            return mood
    return "Unknown Mood"

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize Mediapipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Open webcam
cap = cv2.VideoCapture(0)

prev_emotion = None
prev_sign = None
last_speech_time = 0
speech_delay = 2  # Minimum delay (in seconds) between speech updates

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally
    H, W, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        # Predict emotion
        emotion_prediction = face_model.predict(cropped_img, verbose=0)
        maxindex = int(np.argmax(emotion_prediction))
        detected_emotion = emotion_dict[maxindex]

        if detected_emotion != prev_emotion and time.time() - last_speech_time > speech_delay:
            prev_emotion = detected_emotion
            last_speech_time = time.time()
            tts.say(f"Detected emotion: {detected_emotion}")
            threading.Thread(target=tts.runAndWait).start()

        cv2.putText(frame, detected_emotion, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Hand tracking for sign language detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        if len(results.multi_hand_landmarks) == 1:
            data_aux += [0] * (84 - 42)

        data_aux = np.array(data_aux) / np.max(data_aux)
        input_data = data_aux.reshape(1, 42, 2, 1)

        sign_prediction = sign_model.predict(input_data, verbose=0)
        maxindex = int(np.argmax(sign_prediction))
        detected_sign = sign_classes[maxindex]

        if detected_sign != prev_sign and time.time() - last_speech_time > speech_delay:
            prev_sign = detected_sign
            last_speech_time = time.time()
            tts.say(f"Detected sign: {detected_sign}")
            threading.Thread(target=tts.runAndWait).start()

        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')

        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * W), int(lm.y * H)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, detected_sign, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Face Emotion & Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
