import cv2
import mediapipe as mp
import torch
import pyautogui
import time
import ui_engine as ui # Import from local folder
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from train_model import GestureNetwork 

# --- 1. SETUP ENGINE & PATHS ---
model = GestureNetwork()
model.load_state_dict(torch.load('models/gesture_model.pth')) # Path check
model.eval() 

base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task') # Path check
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    num_hands=1, 
    min_hand_detection_confidence=0.7, 
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

# --- 2. VARS ---
gesture_names = {0: "Fist", 1: "Open Palm", 2: "Pinch", 3: "Idle"}
instructions = [("Fist", "Mute Audio"), ("Open Palm", "Play/Pause"), ("Pinch", "Scroll"), ("Idle", "Tracking")]
pyautogui.FAILSAFE = False 
last_action_time, COOLDOWN, prev_pinch_y, show_info_panel = 0, 2, None, False

# --- 3. WINDOW SETUP (860 x 640) ---
cv2.namedWindow('Hand Telekinesis HUD', cv2.WINDOW_NORMAL) 
cv2.resizeWindow('Hand Telekinesis HUD', 860, 640)

def on_mouse(event, x, y, flags, param):
    global show_info_panel
    if event == cv2.EVENT_MOUSEMOVE:
        show_info_panel = (x < 60 and y < 60) # Smaller hover zone for smaller icon

cv2.setMouseCallback('Hand Telekinesis HUD', on_mouse)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # CRITICAL FIX: Standardize canvas size BEFORE any drawing
    frame = cv2.resize(frame, (860, 640))
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape 
    
    # Process AI
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    display_gesture, display_action, action_color = "Scanning...", "Awaiting Input", ui.TEXT_WHITE 

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        ui.draw_skeleton(frame, landmarks, w, h)

        raw_x = [val for tip in landmarks for val in [tip.x, tip.y]]
        with torch.no_grad(): 
            probs = torch.nn.functional.softmax(model(torch.tensor([raw_x], dtype=torch.float32)), dim=1)[0]
            idx, conf = torch.argmax(probs).item(), probs[torch.argmax(probs)].item()
            
            if conf > 0.85:
                display_gesture = gesture_names[idx]
                curr_t = time.time()
                if idx == 0: # Mute
                    display_action, action_color = "MUTING AUDIO", (50, 50, 255)
                    if curr_t - last_action_time > COOLDOWN:
                        pyautogui.press('volumemute')
                        last_action_time = curr_t
                    prev_pinch_y = None
                elif idx == 1: # Pause
                    display_action, action_color = "PLAY / PAUSE", (255, 200, 0)
                    if curr_t - last_action_time > COOLDOWN:
                        pyautogui.press('playpause')
                        last_action_time = curr_t
                    prev_pinch_y = None
                elif idx == 2: # Scroll
                    display_action, action_color = "SCROLLING", (0, 255, 100)
                    curr_y = landmarks[8].y * h
                    if prev_pinch_y is not None:
                        smoothed = prev_pinch_y + (curr_y - prev_pinch_y) * 0.5
                        if abs(smoothed - prev_pinch_y) > 1.5: 
                            pyautogui.scroll(int((smoothed - prev_pinch_y) * 10))
                        prev_pinch_y = smoothed
                    else: prev_pinch_y = curr_y
                elif idx == 3:
                    display_action, action_color = "IDLE", ui.TEXT_GRAY
                    prev_pinch_y = None
            else: 
                display_gesture, prev_pinch_y = "Transitioning...", None

    ui.draw_liquid_hud(frame, display_gesture, display_action, action_color)
    ui.draw_info_panel(frame, show_info_panel, instructions)

    cv2.imshow('Hand Telekinesis HUD', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()