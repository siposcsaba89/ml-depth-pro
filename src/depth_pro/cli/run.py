#!/usr/bin/env python3
"""Sample script to run DepthPro.

Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""


import argparse
import logging
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import open3d as o3d

from depth_pro import create_model_and_transforms, load_rgb

LOGGER = logging.getLogger(__name__)


def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def save_colored_point_cloud(ply_file, image, depth, K_mat):
   #LOGGER.info(f"Colored point cloud saved to: {ply_file}")
    """Save a colored point cloud to a PLY file."""
    h, w = depth.shape
    fx, fy = K_mat[0, 0], K_mat[1, 1]
    cx, cy = K_mat[0, 2], K_mat[1, 2]

    # Create a point cloud.
    points = []
    colors = []
    for v in range(h):
        for u in range(w):
            z = depth[v, u]
            if z <= 0 or z > 15:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])
            colors.append(image[v, u] / 255.0)

    # Create Open3D point cloud.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save to PLY file.
    o3d.io.write_point_cloud(ply_file, pcd)


def run(args):
    """Run Depth Pro on a sample image."""
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Load model.
    model, transform = create_model_and_transforms(
        device=get_torch_device(),
        precision=torch.half,
    )
    model.eval()

    image_paths = [args.image_path]
    if args.image_path.is_dir():
        image_paths = args.image_path.glob("**/*")
        relative_path = args.image_path
    else:
        relative_path = args.image_path.parent

    if not args.skip_display:
        plt.ion()
        fig = plt.figure()
        ax_rgb = fig.add_subplot(121)
        ax_disp = fig.add_subplot(122)


    for image_path in tqdm(image_paths):
        # Load image and focal length from exif info (if found.).
        cx = None
        cy = None
        try:
            LOGGER.info(f"Loading image {image_path} ...")
            image, _, f_px = load_rgb(image_path)
#            f_px = np.float32(1490.6609)
#            cx = 1441.2617
#            cy = 948.5068
        except Exception as e:
            LOGGER.error(str(e))
            continue
        # Run prediction. If `f_px` is provided, it is used to estimate the final metric depth,
        # otherwise the model estimates `f_px` to compute the depth metricness.
        prediction = model.infer(transform(image), f_px=f_px)

        # Extract the depth and focal length.
        depth = prediction["depth"].detach().cpu().numpy().squeeze()
        if f_px is not None:
            LOGGER.debug(f"Focal length (from exif): {f_px:0.2f}")
        elif prediction["focallength_px"] is not None:
            focallength_px = prediction["focallength_px"].detach().cpu().item()
            f_px = focallength_px
            LOGGER.info(f"Estimated focal length: {focallength_px}")

        inverse_depth = 1 / depth
        # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
        )

        # Save Depth as npz file.
        if args.output_path is not None:
            output_file = (
                args.output_path
                / image_path.relative_to(relative_path).parent
                / image_path.stem
            )
            LOGGER.info(f"Saving depth map to: {str(output_file)}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_file, depth=depth)

            # Save as color-mapped "turbo" jpg image.
            cmap = plt.get_cmap("turbo")
            color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(
                np.uint8
            )
            color_map_output_file = str(output_file) + ".jpg"
            LOGGER.info(f"Saving color-mapped depth to: : {color_map_output_file}")
            PIL.Image.fromarray(color_depth).save(
                color_map_output_file, format="JPEG", quality=90
            )

            # save to colored point cloud in ply format.
            # fx_kitti = 7.188560000000e+02;
            # fy_kitti = 7.188560000000e+02;
            # cx_kitti = 6.071928000000e+02;
            # cy_kitti = 1.852157000000e+02;
            fx_kitti = f_px
            fy_kitti = f_px
            cx_kitti = cx if cx is not None else image.shape[1] / 2
            cy_kitti = cy if cy is not None else image.shape[0] / 2

            K_mat = np.array(
                [
                    [fx_kitti, 0, cx_kitti],
                    [0, fy_kitti, cy_kitti],
                    [0, 0, 1],
                ]
            )
            ply_output_file = str(output_file) + ".ply"
            LOGGER.info(f"Saving colored point cloud to: {ply_output_file}")
            print(f"Image shape: {image.shape}")
            print(f"Depth shape: {depth.shape}")
            print(f"K matrix: {K_mat}")
            save_colored_point_cloud(ply_output_file, image, depth, K_mat)

        # Display the image and estimated depth map.
        if not args.skip_display:
            ax_rgb.imshow(image)
            ax_disp.imshow(inverse_depth_normalized, cmap="turbo")
            fig.canvas.draw()
            fig.canvas.flush_events()

    LOGGER.info("Done predicting depth!")
    if not args.skip_display:
        plt.show(block=True)


def main():
    """Run DepthPro inference example."""
    parser = argparse.ArgumentParser(
        description="Inference scripts of DepthPro with PyTorch models."
    )
    parser.add_argument(
        "-i", 
        "--image-path", 
        type=Path, 
        default="./data/example.jpg",
        help="Path to input image.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        help="Path to store output files.",
    )
    parser.add_argument(
        "--skip-display",
        action="store_true",
        help="Skip matplotlib display.",
    )
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true", 
        help="Show verbose output."
    )
    
    run(parser.parse_args())


if __name__ == "__main__":
    main()
