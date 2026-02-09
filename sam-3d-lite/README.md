# SAM 3D Lite

Multi-view 3D reconstruction using SAM + MiDaS

## Overview

Lightweight 3D reconstruction system that combines:
- **SAM** (Segment Anything Model) for segmentation
- **MiDaS** for depth estimation
- **Epipolar geometry** for multi-view fusion

## Features

- 3D mesh reconstruction from 2-3 images
- No camera calibration required
- Interactive 3D visualization
- ~500 lines of clean Python

## Quick Start

```bash
cd sam-3d-lite
pip install -r requirements.txt
python demo.py --input images/
```

## How It Works

1. Extract SAM masks from multiple views
2. Estimate depth with MiDaS
3. Match features across views
4. Triangulate 3D points
5. Bundle adjustment

## Defense Applications

- UAV surveillance 3D mapping
- Reconnaissance scene reconstruction
- Damage assessment
- Terrain modeling

## Code

See `demo.py` and `sam3d_reconstruction.py` for implementation.
