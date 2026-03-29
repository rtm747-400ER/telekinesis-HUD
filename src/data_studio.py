# This is a script that captures a bunch of images for different hand gestures to create a dataset to train the model

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os

# Prepare the CSV File and Headers
csv_file = 'gesture_dataset.csv'
headers = ['label']
for i in range(21):
    headers.extend([f'x{i}', f'y{i}']) # Generates x0, y0, x1, y1... x20, y20

# Create the file with headers if it doesn't exist yet
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

# Setup the Brain 
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

# HUD Variables
cap = cv2.VideoCapture(0)
recording = False
current_label = -1
frames_recorded = 0
MAX_FRAMES = 500 # How many rows of data to collect per gesture

gesture_names = {0: "Fist", 1: "Open Palm", 2: "Pinch", 3: "Idle"}

print("--- DATA STUDIO READY ---")
print("Press '0' to record Fist")
print("Press '1' to record Open Palm")
print("Press '2' to record Pinch")
print("Press '3' to record an idle hand")
print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    result = detector.detect(mp_image)

    # Base UI Text
    cv2.putText(frame, "Data Collection Studio", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0] # Grab the first hand detected
        h, w, _ = frame.shape
        
        # Draw the full 21-point skeleton so you know the AI sees you
        for tip in landmarks:
            cx, cy = int(tip.x * w), int(tip.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # The Data Extraction Logic
        if recording and frames_recorded < MAX_FRAMES:
            # Start the row with the integer label (0, 1, or 2)
            row = [current_label]
            
            # Extract all 42 coordinates (21 X's and 21 Y's)
            for tip in landmarks:
                row.extend([tip.x, tip.y]) 
            
            # Append this specific frame's data to the CSV
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            frames_recorded += 1
            
            # UI Visual Feedback
            cv2.putText(frame, f"RECORDING {gesture_names[current_label]}: {frames_recorded}/{MAX_FRAMES}", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frame, (0,0), (w,h), (0, 0, 255), 5)
            
        elif recording and frames_recorded >= MAX_FRAMES:
            recording = False # Auto-stop the recording when we hit 500
            print(f"Finished recording 500 frames of {gesture_names[current_label]}!")

    cv2.imshow('Data Studio HUD', frame)
    
    # The Keyboard State Machine
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('0'):
        current_label = 0
        recording = True
        frames_recorded = 0
    elif key == ord('1'):
        current_label = 1
        recording = True
        frames_recorded = 0
    elif key == ord('2'):
        current_label = 2
        recording = True
        frames_recorded = 0
    elif key == ord('3'):
        current_label = 3
        recording = True
        frames_recorded = 0

cap.release()
cv2.destroyAllWindows()