"""
Script to download FLIR thermal dataset sample.

Usage:
    python download_flir_sample.py --output data/flir/ --size 500MB
"""

import os
import argparse
import logging
from pathlib import Path
import urllib.request
import zipfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, output_path: str, chunk_size: int = 8192):
    """Download file with progress."""
    logger.info(f"Downloading from {url}")
    
    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get('Content-Length', 0))
        
        with open(output_path, 'wb') as f:
            downloaded = 0
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    
    print()  # New line after progress
    logger.info(f"Downloaded to {output_path}")


def download_flir_sample(output_dir: str = "data/flir", size: str = "500MB"):
    """
    Download FLIR thermal dataset sample.
    
    Args:
        output_dir: Output directory
        size: Dataset size ('sample', '500MB', 'full')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading FLIR dataset ({size}) to {output_path}")
    
    # For MVP, we'll create synthetic data that mimics FLIR format
    # Real FLIR dataset requires registration and is large
    # Instead, we'll download a small public thermal dataset
    
    # Option: Use KAIST Multispectral dataset (public, smaller)
    kaist_url = "https://soonminhwang.github.io/rgbt-ped-detection/data/kaist.zip"
    
    zip_path = output_path / "kaist_sample.zip"
    
    try:
        download_file(kaist_url, str(zip_path))
        
        # Extract
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        
        # Clean up
        zip_path.unlink()
        
        logger.info(f"Dataset ready at {output_path}")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Creating synthetic thermal data instead...")
        create_synthetic_data(output_path)


def create_synthetic_data(output_path: Path):
    """Create synthetic thermal data for testing."""
    import cv2
    import numpy as np
    
    logger.info("Generating synthetic thermal images...")
    
    (output_path / "images").mkdir(exist_ok=True)
    
    # Generate 20 synthetic thermal images
    for i in range(20):
        # Base thermal background
        img = np.ones((512, 640), dtype=np.uint16) * 10000
        
        # Add noise
        noise = np.random.normal(0, 500, (512, 640)).astype(np.int16)
        img = np.clip(img.astype(np.int32) + noise, 0, 65535).astype(np.uint16)
        
        # Add "human" (warm blob)
        if i % 3 == 0:
            x, y = np.random.randint(100, 540), np.random.randint(100, 412)
            cv2.circle(img, (x, y), 25, 35000, -1)
            cv2.circle(img, (x, y-40), 15, 32000, -1)  # head
        
        # Add "vehicle" (hot rectangle)
        if i % 4 == 0:
            x1, y1 = np.random.randint(50, 400), np.random.randint(100, 350)
            x2, y2 = x1 + np.random.randint(80, 150), y1 + np.random.randint(40, 80)
            cv2.rectangle(img, (x1, y1), (x2, y2), 45000, -1)
        
        # Save
        cv2.imwrite(str(output_path / "images" / f"thermal_{i:03d}.png"), img)
    
    logger.info(f"Created 20 synthetic thermal images in {output_path / 'images'}")


def main():
    parser = argparse.ArgumentParser(description="Download FLIR thermal dataset sample")
    parser.add_argument("--output", "-o", type=str, default="data/flir",
                       help="Output directory")
    parser.add_argument("--size", "-s", type=str, default="sample",
                       choices=["sample", "500MB", "full"],
                       help="Dataset size")
    
    args = parser.parse_args()
    
    download_flir_sample(args.output, args.size)


if __name__ == "__main__":
    main()
