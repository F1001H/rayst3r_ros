import pyzed.sl as sl
import numpy as np
import torch  # Import the torch library
import sys

# ==========================================================
#                >> USER CONFIGURATION <<
# ==========================================================

# 1. IMPORTANT: Set the resolution to match your image capture resolution.
CAMERA_RESOLUTION = sl.RESOLUTION.HD720

# 2. Define the name for the output PyTorch tensor file.
OUTPUT_FILENAME = "zed_intrinsics.pt"

# ==========================================================


def main():
    """
    Connects to a ZED camera, retrieves the intrinsic parameters,
    converts them to a PyTorch tensor, and saves them to a .pt file.
    """
    print("--- ZED Camera Intrinsics Saver (PyTorch .pt format) ---")

    # --- 1. Initialize Camera ---
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = CAMERA_RESOLUTION
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {repr(err)}")
        sys.exit(1)

    print(f"ZED camera opened successfully at {CAMERA_RESOLUTION.name} resolution.")

    # --- 2. Get Camera Information ---
    cam_info = zed.get_camera_information()
    calib_params = cam_info.camera_configuration.calibration_parameters.left_cam

    # --- 3. Assemble the Intrinsics Matrix (K) using NumPy ---
    fx = calib_params.fx
    fy = calib_params.fy
    cx = calib_params.cx
    cy = calib_params.cy

    intrinsics_matrix_np = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)

    # --- 4. Convert NumPy array to PyTorch Tensor ---
    intrinsics_tensor = torch.from_numpy(intrinsics_matrix_np)

    print("\n--- Retrieved Intrinsics Tensor (K) ---")
    print(intrinsics_tensor)
    print(f"Tensor data type: {intrinsics_tensor.dtype}")

    # --- 5. Save the Tensor to a .pt file ---
    try:
        torch.save(intrinsics_tensor, OUTPUT_FILENAME)
        print(f"\nSuccessfully saved intrinsics tensor to '{OUTPUT_FILENAME}'")
    except Exception as e:
        print(f"\nError saving file: {e}")

    # --- 6. Clean Up ---
    zed.close()
    print("Camera closed.")


if __name__ == "__main__":
    main()