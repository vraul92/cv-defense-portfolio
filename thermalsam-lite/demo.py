"""
ThermalSAM-Lite Demo Script

Run detection on single image or batch of thermal images.

Usage:
    python demo.py --input samples/thermal_001.jpg --output output/
    python demo.py --input data/flir/ --output output/ --batch
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from thermal_sam import ThermalSAM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_weights():
    """Download SAM checkpoint if not exists."""
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    checkpoint_path = weights_dir / "sam_vit_b_01ec64.pth"
    
    if checkpoint_path.exists():
        logger.info(f"Checkpoint already exists: {checkpoint_path}")
        return str(checkpoint_path)
    
    logger.info("Downloading SAM ViT-B checkpoint...")
    import urllib.request
    
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    
    try:
        urllib.request.urlretrieve(url, checkpoint_path)
        logger.info(f"Downloaded to {checkpoint_path}")
        return str(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        logger.info("Please download manually from:")
        logger.info("https://github.com/facebookresearch/segment-anything#model-checkpoints")
        sys.exit(1)


def process_single_image(model: ThermalSAM, 
                         image_path: str, 
                         output_dir: str,
                         visualize_modes: list = None):
    """Process a single thermal image."""
    if visualize_modes is None:
        visualize_modes = ["combined"]
    
    logger.info(f"Processing: {image_path}")
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return
    
    logger.info(f"Image shape: {image.shape}, dtype: {image.dtype}")
    
    # Detect objects
    results = model.detect(image, min_area=100)
    
    logger.info(f"Found {len(results)} objects:")
    for i, r in enumerate(results[:5]):  # Show top 5
        logger.info(f"  {i+1}. {r.class_name}: conf={r.confidence:.3f}, "
                   f"temp={r.temperature:.1f}°C, area={r.area}px")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    base_name = Path(image_path).stem
    
    for mode in visualize_modes:
        output_file = output_path / f"{base_name}_{mode}.jpg"
        viz = model.visualize(image, results, mode=mode, save_path=str(output_file))
        logger.info(f"Saved {mode} visualization to {output_file}")
    
    return results


def process_batch(model: ThermalSAM,
                  input_dir: str,
                  output_dir: str,
                  visualize_modes: list = None):
    """Process all images in a directory."""
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    
    # Find all images
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images")
    
    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            process_single_image(model, str(image_file), output_dir, visualize_modes)
        except Exception as e:
            logger.error(f"Error processing {image_file}: {e}")
            continue


def create_sample_image(output_path: str = "samples/thermal_sample.jpg"):
    """Create a synthetic thermal image for testing."""
    logger.info("Creating sample thermal image...")
    
    Path(output_path).parent.mkdir(exist_ok=True)
    
    # Create synthetic thermal image (512x512)
    h, w = 512, 512
    thermal = np.ones((h, w), dtype=np.uint16) * 10000  # Base temperature
    
    # Add "human" signature (warm)
    cv2.circle(thermal, (200, 200), 30, 35000, -1)
    cv2.circle(thermal, (200, 200), 20, 40000, -1)
    
    # Add "vehicle" signature (hot)
    cv2.rectangle(thermal, (350, 150), (450, 250), 50000, -1)
    
    # Add "background" variations
    noise = np.random.normal(0, 1000, (h, w)).astype(np.uint16)
    thermal = cv2.add(thermal, noise)
    
    # Save
    cv2.imwrite(output_path, thermal)
    logger.info(f"Created sample image: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="ThermalSAM-Lite Demo")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input image or directory")
    parser.add_argument("--output", "-o", type=str, default="output/",
                       help="Output directory")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                       help="Path to SAM checkpoint (auto-download if not specified)")
    parser.add_argument("--model-type", "-m", type=str, default="vit_b",
                       choices=["vit_b", "vit_l", "vit_h"],
                       help="SAM model type")
    parser.add_argument("--device", "-d", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device for inference")
    parser.add_argument("--batch", "-b", action="store_true",
                       help="Process directory in batch mode")
    parser.add_argument("--viz-modes", "-v", nargs="+", 
                       default=["combined", "heatmap"],
                       help="Visualization modes")
    parser.add_argument("--create-sample", action="store_true",
                       help="Create a sample thermal image and exit")
    
    args = parser.parse_args()
    
    # Create sample if requested
    if args.create_sample:
        create_sample_image()
        return
    
    # Download weights if needed
    if args.checkpoint is None:
        checkpoint = download_weights()
    else:
        checkpoint = args.checkpoint
    
    # Initialize model
    logger.info("Initializing ThermalSAM...")
    model = ThermalSAM(
        checkpoint_path=checkpoint,
        model_type=args.model_type,
        device=args.device
    )
    
    # Process
    if args.batch or Path(args.input).is_dir():
        process_batch(model, args.input, args.output, args.viz_modes)
    else:
        process_single_image(model, args.input, args.output, args.viz_modes)
    
    logger.info("Done!")


if __name__ == "__main__":
    import numpy as np
    main()
