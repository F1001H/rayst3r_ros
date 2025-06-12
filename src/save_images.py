import pyzed.sl as sl
import numpy as np
import cv2
import os
import sys
import time

def main():
    """
    Main function to capture and save RGB and Depth images from a ZED camera,
    including a warm-up loop to prevent black images.
    """
    # --- 1. Create a ZED camera object ---
    zed = sl.Camera()

    # --- 2. Set initialization parameters ---
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 0
    init_params.depth_maximum_distance = 10.0

    # --- 3. Open the camera ---
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {repr(err)}")
        zed.close()
        sys.exit(1)

    print("ZED camera opened successfully.")
    
    # --- 4. Prepare for capture ---
    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    # =========================================================================
    # NEW: WARM-UP LOOP
    # Grab a few frames and discard them to allow the camera's auto-exposure
    # and auto-white-balance to adjust to the scene.
    # =========================================================================
    print("Warming up the camera...")
    warm_up_frames = 30  # Let's grab 30 frames to be safe
    for i in range(warm_up_frames):
        zed.grab(runtime_parameters)
        # Optional: Add a small delay if needed, though grab() is often enough
        # time.sleep(0.01)
    print("Warm-up complete.")
    # =========================================================================

    # --- 5. Capture the final, good frame ---
    print("Attempting to capture the final frame...")
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # A new, properly exposed frame is available.
        zed.retrieve_image(image_mat, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

        # --- 6. Process and save the images ---
        rgb_image_bgra = image_mat.get_data()
        rgb_image_bgr = rgb_image_bgra[:, :, :3]

        depth_image_meters = depth_mat.get_data()
        depth_image_meters = np.nan_to_num(depth_image_meters, nan=0, posinf=0, neginf=0)
        depth_image_mm_uint16 = (depth_image_meters * 1000).astype(np.uint16)

        # -- Save to disk --
        output_dir = "zed_output"
        os.makedirs(output_dir, exist_ok=True)

        rgb_filename = os.path.join(output_dir, "rgb.png")
        depth_filename = os.path.join(output_dir, "depth.png")

        # Check if the image is actually black before saving
        if np.mean(rgb_image_bgr) < 1.0:
            print("WARNING: The captured RGB image is still very dark or black.")
            print("Please check lighting, lens cap, and USB connection.")
        
        cv2.imwrite(rgb_filename, rgb_image_bgr)
        cv2.imwrite(depth_filename, depth_image_mm_uint16)

        print("-" * 30)
        print(f"Successfully saved images to the '{output_dir}' directory.")
        print(f"RGB image saved as: {rgb_filename}")
        print(f"Depth image saved as: {depth_filename}")
        print(f"  - Pixel values in the depth image represent distance in millimeters.")
        print("-" * 30)

    else:
        print("Failed to grab a frame after warm-up.")

    # --- 7. Close the camera ---
    zed.close()
    print("ZED camera closed.")

if __name__ == "__main__":
    main()