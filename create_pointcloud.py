import numpy as np
import torch
from PIL import Image
import open3d as o3d
import os
import sys

# ==========================================================
#                >> USER CONFIGURATION <<
# ==========================================================

# 1. Define the paths to your input files
RGB_PATH = "zed_output/rgb.png"
DEPTH_PATH = "zed_output/depth.png"
MASK_PATH = "zed_output/mask.png"
INTRINSICS_PATH = "zed_output/intrinsics.pt"

# 2. Define the path for the output point cloud file
OUTPUT_PCD_PATH = "segmented_point_cloud.ply"

# ==========================================================


def main():
    """
    Creates a segmented 3D point cloud from RGB, Depth, Mask, and Intrinsics.
    """
    print("--- Starting Point Cloud Creation ---")

    # --- 1. Load all input files ---
    print("Loading input files...")
    # Check if files exist
    for path in [RGB_PATH, DEPTH_PATH, MASK_PATH, INTRINSICS_PATH]:
        if not os.path.exists(path):
            print(f"Error: Input file not found at '{path}'")
            sys.exit(1)

    # Load images and intrinsics
    rgb_img = Image.open(RGB_PATH)
    depth_img = Image.open(DEPTH_PATH)
    mask_img = Image.open(MASK_PATH)
    K_tensor = torch.load(INTRINSICS_PATH)

    # --- 2. Convert to NumPy arrays and pre-process ---
    print("Processing and validating data...")
    rgb_np = np.array(rgb_img)
    depth_np = np.array(depth_img) # This is uint16 in millimeters
    mask_np = np.array(mask_img)
    K = K_tensor.numpy()

    # Sanity check: ensure all images have the same height and width
    if not (rgb_np.shape[:2] == depth_np.shape[:2] == mask_np.shape[:2]):
        print("Error: Image dimensions do not match!")
        print(f"  RGB:   {rgb_np.shape}")
        print(f"  Depth: {depth_np.shape}")
        print(f"  Mask:  {mask_np.shape}")
        sys.exit(1)
        
    height, width = rgb_np.shape[:2]
    print(f"Images loaded with resolution: {width}x{height}")

    # Convert depth from millimeters (uint16) to meters (float32)
    depth_meters = depth_np.astype(np.float32) / 1000.0

    # Normalize RGB colors from [0, 255] to [0.0, 1.0] for Open3D
    colors_normalized = rgb_np.astype(np.float32) / 255.0

    # Convert mask to a boolean array (True for white pixels, False for black)
    # A threshold of >0 is fine if it's a perfect binary mask. 128 is safer.
    mask_bool = mask_np > 128

    # --- 3. Perform deprojection to get 3D points ---
    print("Calculating 3D points from depth map (deprojection)...")
    
    # Extract intrinsic parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create a grid of pixel coordinates (u, v)
    u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))

    # Apply the deprojection formula vectorized over the entire image
    # Z is simply the depth value at each pixel
    Z = depth_meters
    # X = (u - cx) * Z / fx
    X = (u_coords - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    Y = (v_coords - cy) * Z / fy

    # Stack the X, Y, Z coordinates to get a HxWx3 array of 3D points
    points_3d = np.stack((X, Y, Z), axis=-1)

    # --- 4. Filter points and colors using the segmentation mask ---
    print("Filtering points and colors using the segmentation mask...")
    
    # Use the boolean mask to select only the relevant points and colors
    segmented_points = points_3d[mask_bool]
    segmented_colors = colors_normalized[mask_bool]

    # Handle case where mask is empty
    if len(segmented_points) == 0:
        print("Warning: The provided mask is empty. No points to create. Exiting.")
        sys.exit(0)
        
    print(f"Found {len(segmented_points)} valid points in the mask.")

    # --- 5. Create and save the Open3D point cloud ---
    print("Creating Open3D point cloud object...")
    pcd = o3d.geometry.PointCloud()
    
    # Assign the points and colors to the PointCloud object
    pcd.points = o3d.utility.Vector3dVector(segmented_points)
    pcd.colors = o3d.utility.Vector3dVector(segmented_colors)

    print(f"Saving point cloud to '{OUTPUT_PCD_PATH}'...")
    o3d.io.write_point_cloud(OUTPUT_PCD_PATH, pcd)

    print("\n--- Point cloud creation complete! ---")
    
    # --- 6. (Optional) Visualize the result ---
    print("Displaying the generated point cloud. Close the window to exit.")
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()