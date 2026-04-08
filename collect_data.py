import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os

# --- 1. SETUP THE CSV FILE ---
csv_file = 'hand_data.csv'

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        headers = ['Letter'] + [f'v_{i}' for i in range(1, 64)]
        writer.writerow(headers)

# --- 2. SET UP MEDIAPIPE ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# --- 3. AUTO-COLLECTION VARIABLES (NEW) ---
# Change this number to collect more or less data per letter!
TARGET_FRAMES = 50

is_recording = False
recording_letter = ""
frame_counter = 0

# --- 4. START WEBCAM ---
cap = cv2.VideoCapture(0)
print("--- TURBO DATA COLLECTION MODE ---")
print(f"Press any key (A-Z) to automatically record {TARGET_FRAMES} frames for that letter.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = detector.detect(mp_image)

    # --- 5. THE RECORDING LOGIC ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
        
    # TRIGGER: If we aren't recording, and you press a letter key (A-Z)
    elif not is_recording and ord('a') <= key <= ord('z'):
        is_recording = True
        recording_letter = chr(key).upper()
        frame_counter = 0
        print(f"Started auto-recording for '{recording_letter}'...")

    # ACTION: If recording is active AND a hand is visible
    if is_recording:
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Save the data to CSV
            row_data = [recording_letter]
            for landmark in hand_landmarks:
                row_data.extend([landmark.x, landmark.y, landmark.z])
                
            with open(csv_file, mode='a', newline='') as f:
                csv.writer(f).writerow(row_data)
                
            frame_counter += 1
            
            # Draw a Red progress tracker on the screen
            cv2.putText(frame, f'Recording {recording_letter}: {frame_counter}/{TARGET_FRAMES}', 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Stop recording when we hit the target
            if frame_counter >= TARGET_FRAMES:
                is_recording = False
                print(f"Successfully saved {TARGET_FRAMES} examples for '{recording_letter}'!")
        else:
            # Pause if the hand leaves the frame
            cv2.putText(frame, 'Show Hand to Record!', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
    else:
        # Draw a Green standby message
        cv2.putText(frame, 'Press A-Z to start burst record', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Turbo Data Collection', frame)

cap.release()
cv2.destroyAllWindows()