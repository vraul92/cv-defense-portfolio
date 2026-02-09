#!/usr/bin/env python3
"""
SAM 3D Lite - Demo Script
Multi-view 3D reconstruction using SAM + MiDaS
"""

from sam3d_reconstruction import SAM3DReconstructor, demo_reconstruction
import argparse

def main():
    parser = argparse.ArgumentParser(description="SAM 3D Lite - 3D Reconstruction")
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        help="Input image paths"
    )
    parser.add_argument(
        "--output", "-o",
        default="output.ply",
        help="Output point cloud file (default: output.ply)"
    )
    parser.add_argument(
        "--create-synthetic",
        action="store_true",
        help="Create synthetic test images"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open 3D visualization"
    )
    
    args = parser.parse_args()
    
    if args.create_synthetic or not args.input:
        demo_reconstruction()
    else:
        # Process provided images
        from pathlib import Path
        image_paths = [Path(p) for p in args.input]
        
        print("Initializing SAM 3D...")
        reconstructor = SAM3DReconstructor()
        
        print(f"Processing {len(image_paths)} images...")
        pcd = reconstructor.reconstruct_3d(image_paths, args.output)
        
        if args.visualize:
            reconstructor.visualize(pcd)
        
        print(f"\n✓ Complete! Output: {args.output}")

if __name__ == "__main__":
    main()
