# CV Defense Portfolio

**Expert-Level Computer Vision Projects for Defense Applications**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A curated collection of production-grade computer vision systems designed for defense and surveillance applications. Each project demonstrates expertise in real-time processing, edge deployment, and mission-critical reliability.

**Built for:** Tonbo Imaging Vision & Deep Learning Engineer Role

![Portfolio Overview](https://github.com/vraul92/cv-defense-portfolio/raw/main/assets/portfolio_overview.png)

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
python demo.py --create-sample
python demo_sam3d.py --create-synthetic  # Multi-view 3D reconstruction
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
**Visual Odometry and Trajectory Mapping for GPS-Denied Navigation**

Estimates camera motion from video sequences and reconstructs flight paths. Designed for UAV navigation in GPS-denied environments.

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

# SAM 3D Reconstruction
python thermalsam-lite/demo_sam3d.py --create-synthetic

# Ghost-Tracker
python ghost-tracker/demo_video.py --input your_video.mp4

# Recon-Map
python recon-map/demo.py --input your_video.mp4
```

---

## 🎯 Technical Highlights

| Project | Core Algorithm | Lines of Code | Dependencies |
|---------|---------------|---------------|--------------|
| ThermalSAM-Lite | SAM + Thermal Adaptation | ~1,300 | PyTorch, OpenCV |
| SAM 3D Lite | Multi-View Reconstruction | ~500 | PyTorch, MiDaS |
| Ghost-Tracker | EKF + Hungarian Algorithm | ~900 | NumPy, OpenCV, SciPy |
| Recon-Map | Visual Odometry (ORB) | ~600 | OpenCV, Matplotlib |

**Total Codebase:** ~3,300 lines of production-grade Python

---

## 🏗️ Architecture Overview

### ThermalSAM-Lite
```
Thermal Input (16-bit) → False-Color Mapping → SAM Encoder → 
Mask Decoder → Post-Processing → Visualization (Bbox + Temperature)
```

### SAM 3D Lite
```
Multi-View Images → SAM Segmentation → MiDaS Depth → 
Back-Projection → Point Cloud Fusion → Voxel Downsampling → 3D Output
```

### Ghost-Tracker
```
Video Frame → YOLO Detection → Data Association (Mahalanobis) → 
EKF Update → Ghost Mode Logic → Visualization (Bbox + Ellipse)
```

### Recon-Map
```
Frame → ORB Features → Feature Matching → Essential Matrix → 
Pose Recovery → Trajectory Update → 2D/3D Visualization
```

---

## 🛡️ Defense Relevance

### Why These Projects?

**ThermalSAM-Lite** demonstrates:
- Foundation model adaptation (SAM)
- Domain transfer (RGB → Thermal)
- Edge deployment optimization
- Temperature-aware detection

**SAM 3D Lite** demonstrates:
- Multi-view geometry
- Depth estimation integration
- 3D reconstruction from 2D
- Point cloud processing

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
| SAM 3D Lite | ~1.0 FPS | 6GB | ✓ (batch processing) |
| Ghost-Tracker | ~30 FPS | 2GB | ✓ (real-time) |
| Recon-Map | ~15 FPS | 1GB | ✓ (real-time) |

---

## 📖 Documentation

Each project includes:
- Comprehensive README with usage examples
- Inline code documentation
- Configuration files (YAML)
- Sample data generators
- Demo scripts

### API Examples

**ThermalSAM:**
```python
from thermalsam_lite import ThermalSAM

model = ThermalSAM(checkpoint="sam_vit_b.pth")
results = model.detect(thermal_image)
model.visualize(results, save_path="output.jpg")
```

**Ghost-Tracker:**
```python
from ghost_tracker import GhostTracker

tracker = GhostTracker(max_age=30, min_hits=3)
tracks = tracker.update(detections)
```

**Recon-Map:**
```python
from recon_map import VisualOdometry

vo = VisualOdometry(focal_length=800.0)
pose = vo.process_frame(frame)
trajectory = vo.get_trajectory()
```

---

## 📚 References & Citations

### Core Papers

**SAM (Segment Anything):**
```bibtex
@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C and Lo, Wan-Yen and others},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}
```

**MiDaS (Depth Estimation):**
```bibtex
@article{ranftl2022towards,
  title={Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
  author={Ranftl, René and Bochkovskiy, Alexey and Koltun, Vladlen},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022}
}
```

**Kalman Filter:**
```bibtex
@article{welch1995introduction,
  title={An Introduction to the Kalman Filter},
  author={Welch, Greg and Bishop, Gary},
  year={1995}
}
```

**Visual Odometry:**
```bibtex
@book{ma2004invitation,
  title={An Invitation to 3-D Vision: From Images to Geometric Models},
  author={Ma, Yi and Soatto, Stefano and Kosecka, Jana and Sastry, S. Shankar},
  year={2004},
  publisher={Springer}
}
```

**ORB Features:**
```bibtex
@article{rublee2011orb,
  title={ORB: An efficient alternative to SIFT or SURF},
  author={Rublee, Ethan and Rabaud, Vincent and Konolige, Kurt and Bradski, Gary},
  journal={IEEE International Conference on Computer Vision (ICCV)},
  year={2011}
}
```

---

## 🎓 Learning Path

### For Thermal Imaging:
1. Study SAM architecture (ViT encoder + prompt encoder + mask decoder)
2. Understand thermal-to-RGB domain adaptation
3. Experiment with false-color mapping techniques
4. Practice temperature calibration

### For Tracking:
1. Learn Kalman Filter theory (prediction + update)
2. Understand Mahalanobis distance for data association
3. Implement Hungarian algorithm for assignment
4. Study occlusion handling strategies

### For Visual Odometry:
1. Study epipolar geometry
2. Understand feature extraction (ORB/SIFT)
3. Learn Essential matrix decomposition
4. Practice Bundle Adjustment

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] TensorRT optimization for Jetson
- [ ] Quantization (INT8) for edge deployment
- [ ] Multi-frame temporal consistency
- [ ] ROS2 integration
- [ ] Custom training on defense-specific data

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file.

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

**Built with ❤️ for defense applications.**

*If you find this work useful, please star the repo!* ⭐

---

## 🙏 Acknowledgments

- Meta AI for the [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- Intel for [MiDaS depth estimation](https://github.com/isl-org/MiDaS)
- OpenCV community for computer vision tools
- FLIR Systems for thermal imaging datasets
