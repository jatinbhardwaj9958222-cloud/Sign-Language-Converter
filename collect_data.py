import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os

# --- 1. SETUP THE CSV FILE ---
# This creates a file called 'hand_data.csv' to store our math
csv_file = 'hand_data.csv'

# If the file doesn't exist, we create it and add the column headers
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # First column is the 'Letter', the next 63 are the x,y,z coords for 21 joints
        headers = ['Letter'] + [f'v_{i}' for i in range(1, 64)]
        writer.writerow(headers)

# --- 2. SET UP MEDIAPIPE ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# --- 3. START WEBCAM ---
cap = cv2.VideoCapture(0)
print("--- DATA COLLECTION MODE ---")
print("Hold up a sign (like 'A' or 'B').")
print("Press 'a' on your keyboard to save that frame as the letter A.")
print("Press 'b' to save as B, etc. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = detector.detect(mp_image)

    # Draw a green box in the corner to show the camera is running
    cv2.putText(frame, 'Ready to record...', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Data Collection', frame)

    # --- 4. RECORD THE DATA ---
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    # If a hand is on screen AND you press a letter key (a-z)
    elif detection_result.hand_landmarks and ord('a') <= key <= ord('z'):
        hand_landmarks = detection_result.hand_landmarks[0]
        
        # Convert the pressed key into an uppercase string (e.g., 'a' -> 'A')
        letter_pressed = chr(key).upper()
        
        # Extract the x, y, and z coordinates for all 21 joints
        row_data = [letter_pressed]
        for landmark in hand_landmarks:
            row_data.extend([landmark.x, landmark.y, landmark.z])
            
        # Save the row to our CSV file
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
            
        print(f"Saved 1 example for letter: {letter_pressed}")

cap.release()
cv2.destroyAllWindows()