import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyttsx3
import pickle
import numpy as np

# --- 1. LOAD THE AI BRAIN ---
print("Loading AI Model...")
with open('sign_language_model.pkl', 'rb') as f:
    model = pickle.load(f)

# --- 2. INITIALIZE TEXT-TO-SPEECH ---
engine = pyttsx3.init()
engine.setProperty('rate', 150) 

# --- 3. SET UP MEDIAPIPE ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

def draw_landmarks(image, landmarks):
    h, w, _ = image.shape
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    for start_idx, end_idx in connections:
        start, end = landmarks[start_idx], landmarks[end_idx]
        cv2.line(image, (int(start.x * w), int(start.y * h)), (int(end.x * w), int(end.y * h)), (255, 0, 0), 2)
    for lm in landmarks:
        cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)

# --- 4. START THE WEBCAM ---
cap = cv2.VideoCapture(0)
print("Word Builder Running!")
print("[SPACE] Add Letter | [BACKSPACE] Delete Letter | [S] Speak Word | [Q] Quit")

current_prediction = "Waiting for hand..."
current_word = "" # <--- NEW: This stores the word you are building

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        draw_landmarks(frame, hand_landmarks)

        row_data = []
        for landmark in hand_landmarks:
            row_data.extend([landmark.x, landmark.y, landmark.z])
        
        prediction = model.predict([row_data])
        current_prediction = prediction[0]

        # Draw the live letter prediction
        cv2.putText(frame, f'Live Sign: {current_prediction}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        current_prediction = "Waiting for hand..."

    # <--- NEW: Draw the full word being built lower on the screen
    cv2.putText(frame, f': {current_word}', (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Sign Language Live Converter', frame)

    # --- 5. KEYBOARD CONTROLS ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    # Press SPACEBAR (ASCII code 32) to add the current letter to the word
    elif key == 32: 
        if current_prediction != "Waiting for hand...":
            current_word += current_prediction
            
    # Press BACKSPACE (ASCII code 8) to delete the last letter
    elif key == 8: 
        current_word = current_word[:-1]
        
    # Press 's' to speak the whole word, then clear it to start a new one
    elif key == ord('s'):
        if current_word != "":
            engine.say(current_word)
            engine.runAndWait()
            current_word = "" 

cap.release()
cv2.destroyAllWindows()