# ThermalSAM-Lite

**Zero-Shot Object Detection for Thermal Imagery using Segment Anything Model (SAM)**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight, production-ready implementation of Meta's Segment Anything Model (SAM) adapted for thermal imaging applications. Designed for defense and surveillance scenarios where thermal signature detection is critical.

![ThermalSAM Demo](assets/demo_preview.png)

## 🎯 Features

- **Zero-Shot Detection**: No training required - works out of the box
- **Thermal-Optimized**: Specifically tuned for infrared/thermal imagery
- **False-Color Visualization**: Temperature-based color mapping
- **Multi-Object Support**: Detect humans, vehicles, structures simultaneously
- **Edge-Ready**: Optimized for CPU inference (GPU optional)
- **Real-Time Capable**: ~2-3 FPS on CPU for 512x512 images

## 🏗️ Architecture

```
Thermal Input → Preprocessing → SAM Encoder → Mask Decoder → Post-processing → Visualization
     ↓              ↓                ↓              ↓              ↓                ↓
  Grayscale    False Color      ViT-B/H        Masks       Filter by       Bounding
  16-bit       Mapping          Backbone       + Scores    Temperature     Boxes
```

## 📦 Installation

### Prerequisites
- Python 3.10+
- 8GB+ RAM (16GB recommended)
- 2GB free disk space

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/cv-defense-portfolio.git
cd cv-defense-portfolio/thermalsam-lite

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download SAM checkpoint
python scripts/download_weights.py
```

### Dependencies
- `torch` >= 2.0.0
- `torchvision` >= 0.15.0
- `opencv-python` >= 4.8.0
- `segment-anything` (Meta's official)
- `pillow` >= 10.0.0
- `numpy` >= 1.24.0
- `matplotlib` >= 3.7.0
- `tqdm` >= 4.65.0

## 🚀 Quick Start

### 1. Single Image Demo

```bash
python demo.py --input samples/thermal_001.jpg --output output/
```

### 2. Batch Processing

```bash
python demo.py --input data/flir/ --output output/ --batch
```

### 3. Interactive Mode

```python
from src.thermal_sam import ThermalSAM
import cv2

# Initialize
model = ThermalSAM(checkpoint="weights/sam_vit_b_01ec64.pth", model_type="vit_b")

# Load thermal image
thermal_img = cv2.imread("thermal_001.jpg", cv2.IMREAD_GRAYSCALE)

# Detect objects
results = model.detect(
    image=thermal_img,
    temperature_range=(20, 150),  # Celsius
    min_area=100,  # Filter small detections
    confidence_threshold=0.7
)

# Visualize
model.visualize(results, save_path="output_detected.jpg")
```

## 📊 Dataset

### FLIR Thermal Dataset (Sample)

We use a curated subset (~500MB) of the FLIR ADAS dataset:

- **Source**: https://www.flir.com/oem/adas/thermal-dataset/
- **Size**: ~500 images (day/night scenarios)
- **Format**: 16-bit grayscale PNG + RGB aligned
- **Classes**: Person, Car, Bicycle, Dog

### Download

```bash
python scripts/download_flir_sample.py --output data/flir/ --size 500MB
```

## 🎨 Visualization Modes

### 1. False-Color Thermal
![False Color](assets/viz_false_color.png)

### 2. Segmentation Masks
![Segmentation](assets/viz_segmentation.png)

### 3. Bounding Boxes + Labels
![Detection](assets/viz_detection.png)

### 4. Temperature Heatmap
![Heatmap](assets/viz_heatmap.png)

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
model:
  checkpoint: "weights/sam_vit_b_01ec64.pth"
  model_type: "vit_b"  # vit_b, vit_l, vit_h
  device: "cpu"  # cpu or cuda
  
detection:
  temperature_min: 20.0  # °C
  temperature_max: 150.0  # °C
  confidence_threshold: 0.7
  min_area: 100  # pixels
  max_area: 100000  # pixels
  
visualization:
  colormap: "jet"  # jet, hot, inferno, plasma
  alpha: 0.6  # mask transparency
  show_labels: true
  show_confidence: true
```

## 📈 Performance

| Model | Parameters | CPU Speed (i7) | GPU Speed (RTX 3060) | Memory |
|-------|------------|----------------|----------------------|---------|
| SAM-ViT-B | 91M | ~2.5 FPS | ~15 FPS | 4GB |
| SAM-ViT-L | 308M | ~1.0 FPS | ~8 FPS | 8GB |
| SAM-ViT-H | 636M | ~0.5 FPS | ~5 FPS | 16GB |

## 🛡️ Defense Applications

### 1. Night Surveillance
Detect intruders in complete darkness using thermal signatures.

### 2. Search and Rescue
Locate humans in smoke, fog, or foliage where visual cameras fail.

### 3. Perimeter Security
Automated threat detection for military bases and critical infrastructure.

### 4. UAV Payloads
Lightweight detection for drone-mounted thermal cameras.

## 🔬 Technical Details

### Thermal-to-RGB Adaptation

Thermal cameras output 16-bit grayscale (0-65535). We adapt this for SAM (trained on RGB):

```python
# Normalize to 0-255
normalized = (thermal - thermal.min()) / (thermal.max() - thermal.min()) * 255

# Apply false-color mapping for better feature extraction
colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)

# Convert BGR to RGB for SAM
rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
```

### Temperature Calibration

Convert pixel intensity to temperature using camera calibration:

```python
temperature = (pixel_value - offset) / gain + reference_temp
```

Default parameters work with FLIR Lepton 3.5 sensors.

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run integration test
python tests/test_end_to_end.py
```

## 📝 Citation

If you use this work, please cite:

```bibtex
@software{thermalsam_lite,
  author = {Vuppalapati, Rahul},
  title = {ThermalSAM-Lite: Zero-Shot Thermal Object Detection},
  year = {2025},
  url = {https://github.com/vraul92/cv-defense-portfolio}
}

@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C and Lo, Wan-Yen and others},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}
```

## 🤝 Acknowledgments

- Meta AI for the SAM model
- FLIR Systems for the thermal dataset
- OpenCV community

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🎯 Roadmap

- [ ] TensorRT optimization for Jetson
- [ ] Quantization (INT8) for edge deployment
- [ ] Multi-frame temporal consistency
- [ ] Integration with ROS2
- [ ] Custom training on defense-specific data

## 👤 Author

**Rahul Vuppalapati**  
Senior Data Scientist | Computer Vision & Defense AI  
[LinkedIn](https://linkedin.com/in/vrc7) | [GitHub](https://github.com/vraul92)

---

Built for defense applications. Optimized for the edge. 🛡️
