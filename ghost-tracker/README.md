# Ghost-Tracker

**Real-Time Multi-Object Tracking with Occlusion Handling**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance tracking system implementing Extended Kalman Filter (EKF) for robust multi-object tracking in defense and surveillance scenarios. Features "ghost tracking" to maintain object identity during occlusions.

![Ghost-Tracker Demo](assets/ghost_tracker_demo.gif)

## 🎯 Features

- **Extended Kalman Filter**: Full EKF implementation from scratch (no libraries)
- **Occlusion Handling**: Predicts object position when temporarily hidden
- **Ghost Visualization**: Semi-transparent "ghost boxes" during occlusion
- **Uncertainty Ellipses**: Visual representation of tracking confidence
- **Multi-Object Support**: Track multiple targets simultaneously
- **Real-Time Performance**: Optimized for CPU inference

## 🏗️ Architecture

```
Video Frame → Object Detection (YOLO) → Data Association → EKF Update → Visualization
                  ↓                           ↓                ↓              ↓
             Bounding Boxes          Hungarian Alg.    State Estimate   Ghost Boxes
                                                          + Covariance   + Uncertainty
```

## 📦 Installation

```bash
cd ghost-tracker
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dependencies
- `opencv-python` >= 4.8.0
- `numpy` >= 1.24.0
- `scipy` >= 1.11.0
- `ultralytics` (for YOLOv8)

## 🚀 Quick Start

### Demo with Video File

```bash
python demo_video.py --input samples/traffic.mp4 --output output/
```

### Live Webcam Demo

```bash
python demo_webcam.py
```

### Python API

```python
from src.tracker import GhostTracker
import cv2

# Initialize tracker
tracker = GhostTracker(
    max_age=30,      # Frames to keep ghost track
    min_hits=3,      # Minimum detections to confirm track
    iou_threshold=0.3
)

# Process video
cap = cv2.VideoCapture("input.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects (using YOLO or any detector)
    detections = yolo_detector(frame)
    
    # Update tracker
    tracks = tracker.update(detections)
    
    # Visualize
    for track in tracks:
        if track.is_ghost:
            # Draw ghost box (semi-transparent)
            draw_ghost_box(frame, track)
            draw_uncertainty_ellipse(frame, track)
        else:
            # Draw confirmed track
            draw_bbox(frame, track)
```

## 🔬 Technical Details

### Extended Kalman Filter

Our EKF implementation tracks:
- **State**: `[x, y, w, h, dx, dy]` (position, size, velocity)
- **Measurement**: `[x, y, w, h]` (bounding box from detector)
- **Prediction**: Uses constant velocity motion model
- **Update**: Corrects with new detection using Kalman gain

### Occlusion Handling

When an object is not detected:
1. Continue prediction using motion model
2. Increment "ghost" counter
3. Expand uncertainty ellipse (covariance grows)
4. If object reappears within search radius → reassociate
5. If ghost counter exceeds max_age → delete track

### Data Association

Uses Mahalanobis distance for track-to-detection matching:
```
d² = (z - Hx)^T * S^(-1) * (z - Hx)
```

Where:
- `z`: Detection measurement
- `Hx`: Predicted measurement
- `S`: Innovation covariance

## 📊 Performance

| Scenario | FPS (CPU) | Tracks | Occlusion Recovery |
|----------|-----------|--------|-------------------|
| Single Object | 30+ | 1 | 95% |
| Multi-Object (5) | 25+ | 5 | 88% |
| Heavy Occlusion | 20+ | 10+ | 75% |

## 🛡️ Defense Applications

### 1. Perimeter Security
Track intruders across camera zones, maintaining identity during occlusions.

### 2. UAV Tracking
Follow targets from aerial platforms despite clouds, buildings, or terrain occlusion.

### 3. Convoy Protection
Track vehicles in a convoy, handling overlapping and temporary disappearances.

### 4. Missile Guidance
Predict target trajectory during evasive maneuvers or countermeasures.

## 🎨 Visualization

### Ghost Mode
When target is occluded:
- Box becomes semi-transparent (alpha = 0.5)
- Dashed border
- Uncertainty ellipse grows
- Predicted trajectory shown as dotted line

### Confirmed Track
- Solid colored box
- Track ID label
- Velocity vector arrow
- Trail of previous positions

## 🔧 Configuration

Edit `config.yaml`:

```yaml
tracker:
  max_age: 30          # Frames before deleting ghost track
  min_hits: 3          # Detections needed to confirm track
  iou_threshold: 0.3   # For initial association
  
kalman:
  process_noise: 0.05
  measurement_noise: 0.1
  initial_covariance: 1.0

detection:
  model: "yolov8n.pt"  # YOLO model
  confidence: 0.5
  classes: [0]         # COCO class IDs (0=person, 2=car, etc.)
```

## 📝 Citation

```bibtex
@software{ghost_tracker,
  author = {Vuppalapati, Rahul},
  title = {Ghost-Tracker: EKF-based Multi-Object Tracking},
  year = {2025},
  url = {https://github.com/vraul92/cv-defense-portfolio}
}
```

## 👤 Author

**Rahul Vuppalapati**  
Senior Data Scientist | Computer Vision & Tracking Systems  
[LinkedIn](https://linkedin.com/in/vrc7) | [GitHub](https://github.com/vraul92)
