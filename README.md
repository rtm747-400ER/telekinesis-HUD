# 🌌 Telekinesis-HUD
### **A Neural Interface for Kinetic OS Control**
 
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/MediaPipe-00BFFF?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Platform-Windows-0078D4?style=for-the-badge&logo=windows&logoColor=white" />
</p>
 
<p align="center">
  <b>Telekinesis-HUD</b> is a high-performance, real-time computer vision utility that translates complex hand gestures into system-level OS commands with just a webcam and your hands (Accessibility note: Even a single hand will do :D).
</p>
 
<p align="center">
  It features a bespoke <em>frosted-glass</em> HUD built from scratch using OpenCV alpha-blending and Gaussian rendering, delivering zero-latency visual feedback while keeping your workspace clean and unobtrusive.
</p>
 
---
 
## 🎬 Demo
 
> ![Telekinesis-HUD Demo Screenshot](assets/screenshot1.png)

https://github.com/user-attachments/assets/9674f643-012c-45f3-84f4-cccda6ed3a28

*Real-time gesture recognition running at HD resolution. The frosted-glass HUD overlays gesture state and active OS action without obscuring the camera feed.*
 
---
 
## ⚡ Core Features
 
- **🧠 Neural Gesture Engine** — A custom-trained Multi-Layer Perceptron (MLP) built in **PyTorch**, performing real-time inference on a 21-point 3D hand landmark coordinate system extracted per-frame.
 
- **🪟 Frosted Glass HUD** — A bespoke rendering engine built entirely in **OpenCV** using Gaussian blurs, alpha-channel compositing, and anti-aliased geometry to produce a translucent, professional UI aesthetic.
 
- **🎮 Precision OS Control** — Four gesture-to-command mappings with a configurable cooldown system to eliminate accidental triggers:
 
  | Gesture | Action | Feedback Color |
  |:--|:--|:--|
  | ✊ Fist | System-wide Audio Mute / Unmute | 🔴 Red |
  | 🖐️ Open Palm | Media Play / Pause | 🟡 Amber |
  | 🤏 Pinch & Drag | Dynamic Smooth Scrolling (Lerp) | 🟢 Green |
  | ✋ Idle | Passive Tracking — No Action Fired | ⚪ Gray |
 
- **📐 Lerp Scroll Engine** — Pinch-based scrolling uses linear interpolation (`α = 0.5`) and a dead-zone threshold to eliminate jitter and deliver smooth, responsive page control.
 
- **🖱️ Context-Sensitive Info Panel** — Hover the top-left corner to reveal the gesture reference panel. Move away to dismiss. No cluttered UI when you don't need it.
 
- **⚙️ Optimized Performance** — All frames are standardized to an **HD canvas** before inference, balancing visual fidelity with consistent, fast frame rates.
 
---
 
## 🛠️ Tech Stack
 
| Layer | Technology | Role |
|:--|:--|:--|
| **Vision** | Google MediaPipe `hand_landmarker` | Real-time 3D hand landmark extraction (21 keypoints) |
| **Inference** | PyTorch `GestureNetwork` (MLP) | Gesture classification with softmax confidence gating |
| **Graphics** | OpenCV + Custom `ui_engine` | HD rendering, skeleton overlay, frosted-glass HUD |
| **Automation** | PyAutoGUI | System-level HID keystroke injection |
| **Math** | NumPy | Coordinate normalization, Lerp smoothing |
 
---
 
## 🧠 How It Works
 
The pipeline runs end-to-end on every captured frame:
 
```
Webcam Frame
    │
    ▼
[ Resize + Flip ]  ←  Standardize to 860×640, mirror for natural UX
    │
    ▼
[ MediaPipe Hand Landmarker ]  ←  Extract 21 (x, y) 3D keypoints
    │
    ▼
[ Flatten → 42-dim vector ]  ←  Raw coordinate array
    │
    ▼
[ GestureNetwork (MLP) ]  ←  PyTorch forward pass
    │
    ▼
[ Softmax + Confidence Gate (> 0.85) ]  ←  Reject ambiguous frames
    │
    ▼
[ Action Dispatch + Cooldown Check ]  ←  Fire OS command if stable
    │
    ▼
[ HUD Render + Display ]  ←  Overlay gesture state on frame
```
 
> **Confidence Gate:** Any frame where the model's top-class probability is below **0.85** is classified as `"Transitioning..."` and no OS action is fired. This prevents misfires mid-gesture.
 
> **Cooldown System:** Discrete actions (Mute, Play/Pause) are locked behind a **2-second cooldown** to prevent repeated rapid-fire triggers from a held gesture.
 
---
 
## 📂 Project Structure
 
```text
Telekinesis-HUD/
│
├── data/
│   └── gestures.csv          # Raw (x, y) coordinate datasets per gesture class
│
├── models/
│   ├── gesture_model.pth     # Trained MLP weights
│   └── hand_landmarker.task  # MediaPipe hand landmark model (binary)
│
├── src/
│   ├── app.py                # Main entry point — capture loop, OS dispatch, HUD
│   ├── ui_engine.py          # "Liquid Glass" graphics engine — HUD, skeleton, panels
│   ├── train_model.py        # GestureNetwork architecture + training loop
│   └── data_studio.py        # Coordinate collection & labelling utility
│
├── assets/                   # Screenshots, GIFs for README
├── requirements.txt          # Pinned dependencies (generated via pipreqs)
└── README.md
```
 
---
 
## 🚀 Getting Started
 
### Prerequisites
 
- Python **3.9+**
- A webcam (built-in or USB)
- Windows OS *(PyAutoGUI key bindings are Windows-targeted)*
 
### 1. Clone the Repository
 
```bash
git clone https://github.com/YOUR_USERNAME/Telekinesis-HUD.git
cd Telekinesis-HUD
```
 
### 2. Install Dependencies
 
```bash
pip install -r requirements.txt
```
 
> **Note:** If you hit a `mediapipe` install error on Python 3.12+, downgrade to Python 3.11 via pyenv or conda.
 
### 3. Download Model Assets
 
Ensure both model files are present before running:
 
```
models/gesture_model.pth        ← Trained gesture classifier
models/hand_landmarker.task     ← MediaPipe landmark model
```
 
> Download `hand_landmarker.task` from the [MediaPipe Model Zoo](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).
 
### 4. Run
 
```bash
python src/app.py
```
 
Press **`Q`** to quit the HUD window.
 
---
 
## 🏋️ Training Your Own Model
 
If you want to retrain on your own hand shape / environment, the full data pipeline is included.
 
### Step 1 — Collect Gesture Data
 
```bash
python src/data_studio.py
```
 
This opens the webcam and lets you record labeled gesture samples, saved to `data/gestures.csv`. Record at least **500 samples per class** for reliable accuracy.
 
### Step 2 — Train the Network
 
```bash
python src/train_model.py
```
 
The `GestureNetwork` MLP trains on the collected CSV and saves the best weights to `models/gesture_model.pth`.
 
### Model Architecture
 
```
Input Layer   →  42 nodes  (21 landmarks × 2 axes)
Hidden Layer  →  64 nodes  (ReLU)
Hidden Layer  →  32 nodes  (ReLU)
Output Layer  →   4 nodes  (Softmax → Fist, Open Palm, Pinch, Idle)
```
 
> **Tip:** If accuracy on your hand is poor, re-collect data in your exact lighting conditions and webcam angle — the model is highly sensitive to these.
 
---
 
## 🎨 HUD Design System
 
The `ui_engine.py` module is a self-contained rendering library. Key visual components:
 
| Component | Technique |
|:--|:--|
| Frosted glass panels | Gaussian blur + alpha composite over live frame |
| Skeleton overlay | Polyline + circle rendering on MediaPipe landmark graph |
| Action color coding | Per-gesture RGB accent passed through draw calls |
| Info panel | Mouse-hover triggered, renders top-left on `EVENT_MOUSEMOVE` |
| Status text | Anti-aliased `cv2.putText` with drop-shadow pass |
 
---
 
## 🤝 Contributing
 
Contributions, issues, and feature requests are welcome.
 
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request
 
---
 
## 🙏 Acknowledgements
 
- [Google MediaPipe](https://ai.google.dev/edge/mediapipe) — for the hand landmark model
- [PyAutoGUI](https://pyautogui.readthedocs.io/) — for system-level HID control
- [PyTorch](https://pytorch.org/) — for the neural network framework
 
---
 
<p align="center">
  <sub>If this project helped you, consider dropping a ⭐ on the repo.</sub>
</p>
