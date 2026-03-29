# 🌌 Telekinesis-HUD
### **A Neural Interface for Kinetic OS Control**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-00BFFF?logo=google&logoColor=white)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)

**Telekinesis-HUD** is a high-performance, real-time computer vision utility that translates complex hand gestures into system-level Windows OS commands. It features a bespoke frosted-glass HUD that provides zero-latency visual feedback while keeping your workspace clean.

---

## ⚡ Core Features

* **Neural Gesture Engine:** A custom-trained Multi-Layer Perceptron (MLP) built in **PyTorch**, handling real-time inference on a 21-point hand landmark coordinate system.
* **Frosted Glass UI:** Custom-built rendering engine using **OpenCV** Gaussian blurs and alpha-blending to create a professional, translucent aesthetic.
* **Precision OS Control:** 
    * **Fist:** System-wide Audio Mute/Unmute.
    * **Open Palm:** Media Play/Pause.
    * **Pinch & Drag:** Dynamic Smooth Scrolling with Linear Interpolation (Lerp) to eliminate jitter.
* **Optimized Performance:** Running on a standardized **HD Canvas** to balance visual fidelity with blazing-fast frame rates.

---

## 🛠️ The Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Vision** | Google MediaPipe | Real-time 3D hand landmark extraction. |
| **Inference** | PyTorch | Custom MLP model for gesture classification. |
| **Graphics** | OpenCV | HD UI rendering, anti-aliased geometry, and "Liquid Glass" effects. |
| **Automation** | PyAutoGUI | System-level HID (Human Interface Device) hijacking. |
| **Math** | NumPy | Coordinate normalization and vector smoothing. |

---

## 📂 Project Structure

This project follows a modular architecture to keep the neural logic isolated from the rendering engine.

```text
Telekinesis-HUD/
├── data/                 # Raw coordinate datasets (.csv)
├── models/               # Weights (.pth) and hand landmarker tasks
├── src/
│   ├── app.py            # Main entry point & OS logic
│   ├── ui_engine.py      # The "Liquid Glass" graphics engine
│   ├── train_model.py    # Neural network architecture & training
│   └── data_studio.py    # Coordinate collection utility
├── requirements.txt      # Verified dependencies (via pipreqs)
└── README.md             # The face of the project