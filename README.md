# AI Live Object Identifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-World-green)
![License](https://img.shields.io/badge/License-Non--Commercial-orange)

A high-performance, real-time AI object detection application built with Python, OpenCV, and YOLO-World. Designed for versatility, it runs on everything from Raspberry Pis to high-end RTX workstations, featuring a polished UI, open-vocabulary detection, and robust object tracking.

## 🌟 Features

*   **Open-Vocabulary Detection**: Detect *any* object by simply typing its name. Powered by `YOLO-World`.
*   **Cross-Platform Acceleration**: Auto-detects and uses **NVIDIA CUDA**, **Apple Silicon (MPS)**, or **AMD/CPU**.
*   **3 Power Tiers**:
    *   **Low**: `yolov8n` (Raspberry Pi / CPU)
    *   **Medium**: `yolov8s-world` (Laptops)
    *   **High**: `yolov8x-world` (Desktop / RTX)
*   **Professional UI**:
    *   **Zero-Flicker** rendering engine.
    *   **Maximized Window** mode.
    *   **Glassmorphism** status bar and notifications.
    *   **Letterboxing** for cinematic, non-stretched video.
*   **Robust Tracking**: Integrated **BoT-SORT** for stable object ID tracking.
*   **In-App Input**: Type custom classes directly in the GUI without touching the terminal.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Lishen99/ai-live-object-identifier.git
    cd ai-live-object-identifier
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install GPU Support (Recommended)**:
    *   **NVIDIA Users**:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
    *   **Mac / AMD / CPU Users**:
        ```bash
        pip install torch torchvision torchaudio
        ```

## 🚀 Usage

Run the main application:
```bash
python main.py
```

### Controls

| Key | Action | Description |
| :--- | :--- | :--- |
| **1** | **Low Power** | Switch to `yolov8n` (Standard COCO classes). |
| **2** | **Balanced** | Switch to `yolov8s-world` (Open Vocabulary). |
| **3** | **High Power** | Switch to `yolov8x-world` (Best Accuracy). |
| **M** | **Switch Mode** | Cycle: **ALL** -> **LIVING** -> **OBJECTS**. |
| **C** | **Custom Classes** | Open the in-app text box to type new objects to detect. |
| **T** | **Toggle Tracking** | Turn object ID tracking on/off. |
| **Q** | **Quit** | Exit the application. |

## 📂 Project Structure

*   `main.py`: Entry point. Handles the event loop, window management, and user input.
*   `detector.py`: Manages YOLO models, inference, tracking, and class filtering.
*   `visualizer.py`: Handles all drawing, overlays, notifications, and the GUI input box.
*   `camera.py`: Robust video capture wrapper.

## 📝 License

This project is licensed under the Non-Commercial License - see the [LICENSE](LICENSE) file for details.
