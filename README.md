# CV Defense Portfolio

**Expert-Level Computer Vision Projects for Defense Applications**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A curated collection of production-grade computer vision systems designed for defense and surveillance applications. Each project demonstrates expertise in real-time processing, edge deployment, and mission-critical reliability.

**Built for:** Tonbo Imaging Vision & Deep Learning Engineer Role

---

## 📁 Projects

### 1. 🔥 ThermalSAM-Lite
**Zero-Shot Object Detection for Thermal Imagery**

Adapts Meta's Segment Anything Model (SAM) for defense-grade thermal imaging. Detects humans, vehicles, and heat signatures without domain-specific training.

**Key Features:**
- SAM-based zero-shot detection
- Thermal-to-RGB domain adaptation
- False-color visualization
- Temperature calibration
- Edge-optimized inference

```bash
cd thermalsam-lite
python demo.py --input samples/thermal_001.jpg --output output/
```

**Defense Applications:** Night surveillance, search & rescue, perimeter security, UAV payloads

---

### 2. 👻 Ghost-Tracker
**Real-Time Multi-Object Tracking with Occlusion Handling**

Implements Extended Kalman Filter (EKF) from scratch for robust multi-object tracking. Features "ghost mode" to maintain identity during occlusions.

**Key Features:**
- Full EKF implementation (no libraries)
- Mahalanobis distance data association
- Ghost mode for occlusion handling
- Uncertainty ellipse visualization
- Multi-object support

```bash
cd ghost-tracker
python demo_video.py --input samples/traffic.mp4 --output output/
```

**Defense Applications:** Perimeter security, UAV tracking, convoy protection, missile guidance

---

### 3. 🗺️ Recon-Map
**Visual Odometry for GPS-Denied Navigation**

Estimates camera motion from video sequences and reconstructs flight paths. Designed for UAV navigation without GPS signals.

**Key Features:**
- Monocular visual odometry
- ORB feature extraction
- Essential matrix decomposition
- Real-time 2D/3D trajectory mapping
- Interactive dashboard

```bash
cd recon-map
python demo.py --input video.mp4 --output output/
```

**Defense Applications:** UAV navigation, indoor reconnaissance, subterranean operations, urban canyon navigation

---

## 🚀 Quick Start

### Clone and Setup

```bash
git clone https://github.com/vraul92/cv-defense-portfolio.git
cd cv-defense-portfolio

# Each project has its own setup
for project in thermalsam-lite ghost-tracker recon-map; do
    cd $project
    python3.10 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
done
```

### Run All Demos

```bash
# ThermalSAM-Lite
python thermalsam-lite/demo.py --create-sample

# Ghost-Tracker
python ghost-tracker/demo_video.py --input your_video.mp4

# Recon-Map
python recon-map/demo.py --input your_video.mp4
```

---

## 🎯 Technical Highlights

| Project | Core Algorithm | Lines of Code | Dependencies |
|---------|---------------|---------------|--------------|
| ThermalSAM-Lite | SAM + Thermal Adaptation | ~800 | PyTorch, OpenCV |
| Ghost-Tracker | EKF + Hungarian Algorithm | ~900 | NumPy, OpenCV, SciPy |
| Recon-Map | Visual Odometry (ORB) | ~600 | OpenCV, Matplotlib |

**Total Codebase:** ~2,300 lines of production-grade Python

---

## 🛡️ Defense Relevance

### Why These Projects?

**ThermalSAM-Lite** demonstrates:
- Foundation model adaptation (SAM)
- Domain transfer (RGB → Thermal)
- Edge deployment optimization
- Temperature-aware detection

**Ghost-Tracker** demonstrates:
- State estimation (Kalman filtering)
- Data association (Mahalanobis distance)
- Occlusion handling (ghost mode)
- Multi-object tracking

**Recon-Map** demonstrates:
- Geometric computer vision
- Structure from Motion
- GPS-denied navigation
- Real-time pose estimation

---

## 📊 Performance Benchmarks

All projects optimized for **CPU inference** (no GPU required):

| Project | CPU (i7) | Memory | Real-time? |
|---------|----------|--------|------------|
| ThermalSAM-Lite | ~2.5 FPS | 4GB | ✓ (for surveillance) |
| Ghost-Tracker | ~30 FPS | 2GB | ✓ (real-time) |
| Recon-Map | ~15 FPS | 1GB | ✓ (real-time) |

---

## 🎓 Learning Resources

Each project includes:
- Comprehensive README
- Inline code documentation
- Jupyter notebooks for exploration
- Sample data generators
- Unit tests

---

## 👤 Author

**Rahul Vuppalapati**  
Senior Data Scientist | Computer Vision & Defense AI  

**Experience:**
- 8+ years in ML/CV (Apple, Walmart, IBM)
- GenAI & RAG Expert
- Edge AI & Embedded Systems
- Real-time Computer Vision

**Contact:**
- 📧 vrc7.ds@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/vrc7)
- 🐙 [GitHub](https://github.com/vraul92)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file.

---

**Built with ❤️ for defense applications.**

*If you find this work useful, please star the repo!* ⭐
