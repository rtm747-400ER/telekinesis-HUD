import cv2
import numpy as np

# Apple Design Colors (BGR)
APPLE_BLUE = (255, 170, 50) 
TEXT_WHITE = (240, 240, 240)
TEXT_GRAY = (180, 180, 180)

def draw_rounded_rect(img, top_left, bottom_right, color, thickness=1, radius=12):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)

def draw_skeleton(frame, landmarks, w, h):
    joint_coords = [(int(tip.x * w), int(tip.y * h)) for tip in landmarks]
    hand_bones = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
    ]
    for bone in hand_bones:
        cv2.line(frame, joint_coords[bone[0]], joint_coords[bone[1]], (0, 255, 100), 2, cv2.LINE_AA)
    for coord in joint_coords:
        cv2.circle(frame, coord, 4, (255, 255, 255), -1, cv2.LINE_AA)

def draw_liquid_hud(frame, gesture, action, action_color):
    h, w, _ = frame.shape
    hud_height = 110 # Tuned for 640 height
    hud_y = h - hud_height
    roi = frame[hud_y:h, 0:w]
    blurred = cv2.GaussianBlur(roi, (55, 55), 0)
    tint = np.zeros_like(blurred)
    glass = cv2.addWeighted(blurred, 0.6, tint, 0.4, 0)
    frame[hud_y:h, 0:w] = glass
    cv2.line(frame, (0, hud_y), (w, hud_y), (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(frame, "STATUS:", (30, h - 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, gesture.upper(), (30, h - 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "ACTION:", (w - 350, h - 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, action, (w - 350, h - 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, action_color, 2, cv2.LINE_AA)

def draw_info_panel(frame, show_panel, instructions):
    if show_panel:
        p_w, p_h = 420, 240 # Tuned for 860 width
        p_x, p_y = 15, 15
        roi = frame[p_y:p_y+p_h, p_x:p_x+p_w]
        blurred = cv2.GaussianBlur(roi, (55, 55), 0)
        frame[p_y:p_y+p_h, p_x:p_x+p_w] = cv2.addWeighted(blurred, 0.5, np.zeros_like(blurred), 0.5, 0)
        draw_rounded_rect(frame, (p_x, p_y), (p_x+p_w, p_y+p_h), (255, 255, 255), 1, radius=12)
        cv2.putText(frame, "Gesture Guide", (p_x+20, p_y+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, "PRESS 'Q' TO QUIT", (p_x+250, p_y+38), cv2.FONT_HERSHEY_DUPLEX, 0.45, TEXT_GRAY, 1, cv2.LINE_AA)
        cv2.line(frame, (p_x+20, p_y+55), (p_x+p_w-20, p_y+55), (100, 100, 100), 1, cv2.LINE_AA)
        for i, (g, a) in enumerate(instructions):
            ty = p_y + 90 + (i * 35)
            cv2.putText(frame, f"{g}", (p_x+20, ty), cv2.FONT_HERSHEY_DUPLEX, 0.55, APPLE_BLUE, 1, cv2.LINE_AA)
            cv2.putText(frame, a, (p_x + 180, ty), cv2.FONT_HERSHEY_DUPLEX, 0.55, TEXT_WHITE, 1, cv2.LINE_AA)
    else:
        i_s, i_x, i_y = 40, 15, 15
        roi = frame[i_y:i_y+i_s, i_x:i_x+i_s]
        blurred = cv2.GaussianBlur(roi, (25, 25), 0)
        frame[i_y:i_y+i_s, i_x:i_x+i_s] = cv2.addWeighted(blurred, 0.7, np.zeros_like(blurred), 0.3, 0)
        draw_rounded_rect(frame, (i_x, i_y), (i_x+i_s, i_y+i_s), (255, 255, 255), 1, radius=8)
        cv2.putText(frame, "i", (i_x+16, i_y+28), cv2.FONT_HERSHEY_DUPLEX, 0.7, TEXT_WHITE, 1, cv2.LINE_AA)