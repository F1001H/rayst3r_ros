# RaySt3R ROS Integration

This repository provides a ROS (Robot Operating System) wrapper for the [RaySt3R: A Ray-based Multi-view Stereo Network](https://github.com/Duisterhof/rayst3r) project. It allows RaySt3R to consume RGB images, depth maps, and camera intrinsic data from ROS topics and publish the resulting 3D point cloud reconstruction as a `sensor_msgs/PointCloud2` message.

This integration aims to make RaySt3R more accessible for robotics applications where live sensor data is common.

## Features

*   **ROS Topic Input:**
    *   Subscribes to RGB image topics (`sensor_msgs/Image`).
    *   Subscribes to depth image topics (`sensor_msgs/Image`), handling common encodings like `16UC1` (millimeters) and `32FC1` (meters).
    *   Subscribes to camera info topics (`sensor_msgs/CameraInfo`) for intrinsic parameters.
    *   Topic names are configurable via a `camera_config.yaml` file.
*   **Object Segmentation (Optional Prerequisite):**
    *   Includes an object segmentation pipeline using Owl-ViT for object detection based on text prompts and Segment Anything Model (SAM) for mask generation.
    *   This generates a binary mask that can be used by RaySt3R for focused reconstruction.
*   **RaySt3R Inference:**
    *   Utilizes a modified version of the RaySt3R inference pipeline.
    *   The core `GenericLoaderSmall` from RaySt3R is adapted to accept direct NumPy array inputs (RGB, depth, mask, intrinsics, pose) instead of loading from files.
*   **ROS PointCloud2 Output:**
    *   Publishes the reconstructed 3D point cloud as a `sensor_msgs/PointCloud2` message.
    *   The output topic and frame ID are configurable.
*   **Configuration:**
    *   Key parameters like ROS topic names and some pipeline settings can be configured via a `camera_config.yaml` file.