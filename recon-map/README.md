# Recon-Map

**Visual Odometry and Trajectory Mapping for GPS-Denied Navigation**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A real-time visual odometry system that estimates camera motion from video sequences and reconstructs the flight path. Designed for UAV navigation in GPS-denied environments.

![Recon-Map Demo](assets/recon_map_demo.gif)

## 🎯 Features

- **Monocular Visual Odometry**: Estimate motion from single camera
- **Feature Tracking**: ORB/SIFT feature extraction and matching
- **Trajectory Mapping**: Real-time 3D path visualization
- **GPS-Denied Navigation**: No external positioning required
- **Interactive Dashboard**: Live 2D/3D trajectory plots

## 🏗️ Architecture

```
Video Frame → Feature Extraction (ORB) → Feature Matching → 
Motion Estimation (Essential Matrix) → Trajectory Update → Visualization
```

## 📦 Installation

```bash
cd recon-map
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Quick Start

### Process Video

```bash
python demo.py --input video.mp4 --output output/
```

### Live Webcam Demo

```bash
python demo_live.py
```

## 🛡️ Defense Applications

### 1. UAV Navigation
Navigate drones in GPS-jammed environments using visual landmarks.

### 2. Indoor Reconnaissance
Map indoor structures without GPS signals.

### 3. Subterranean Operations
Navigate tunnels and caves using visual odometry.

### 4. Urban Canyon Navigation
Maintain position awareness in dense urban environments.

## 👤 Author

**Rahul Vuppalapati**  
Senior Data Scientist | Computer Vision & Navigation Systems  
[LinkedIn](https://linkedin.com/in/vrc7) | [GitHub](https://github.com/vraul92)
