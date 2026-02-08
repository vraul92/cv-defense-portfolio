"""
Script to download SAM model weights.

Usage:
    python download_weights.py
"""

import os
import urllib.request
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_sam_weights(model_type: str = "vit_b", output_dir: str = "weights"):
    """
    Download SAM model checkpoint.
    
    Args:
        model_type: Model variant (vit_b, vit_l, vit_h)
        output_dir: Directory to save weights
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }
    
    if model_type not in urls:
        raise ValueError(f"Unknown model type: {model_type}")
    
    url = urls[model_type]
    filename = url.split("/")[-1]
    output_file = output_path / filename
    
    if output_file.exists():
        logger.info(f"Weights already exist: {output_file}")
        return str(output_file)
    
    logger.info(f"Downloading SAM {model_type} weights...")
    logger.info(f"URL: {url}")
    logger.info(f"Output: {output_file}")
    
    try:
        urllib.request.urlretrieve(url, output_file)
        logger.info(f"✓ Download complete: {output_file}")
        return str(output_file)
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    model_type = sys.argv[1] if len(sys.argv) > 1 else "vit_b"
    download_sam_weights(model_type)
