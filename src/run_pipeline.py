import yaml
import os
import threading
import sys

import matplotlib.pyplot as plt

# Your project's inference and annotation modules
from annotate_image import run_object_segmentation_pipeline

# MODIFIED: Import the NEW multi-object publisher classes from your driver.
from ros_driver import (
    RosImageSubscriber,
    RosMergedPointCloudPublisher,
    RosInstanceMaskPublisher,
    RosDetection2DArrayPublisher,
)

def load_camera_config(config_filename="camera_config.yaml"):
    """Loads camera configuration from a YAML file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_filename)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if not config:
                raise ValueError("Config file is empty or malformed.")
            return config
    except FileNotFoundError:
        print(f"ERROR: Camera config file not found at {config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load camera config: {e}")
        sys.exit(1)

def run():
    # The first ROS class instantiated will initialize the node.
    print("Starting reconstruction and perception pipeline...")
    
    camera_config = load_camera_config()
    print('Camera config loaded.')
    
    image_sub = RosImageSubscriber(
        rgb_topic=camera_config.get("rgb_topic"),
        depth_topic=camera_config.get("depth_topic"),
        cam_info_topic=camera_config.get("cam_info_topic")
    )
    rgb_data, depth_data_metric, K_data, source_header = image_sub.get_all_data()
    if rgb_data is None:
        print("ERROR: Failed to get initial image data. Exiting.")
        return

    # --- MODIFIED: Handle lists of results from the perception pipeline ---
    print("Processing data to generate masks and bounding boxes for all prompted objects...")
    # The perception pipeline now returns a list of masks and a list of boxes
    all_masks, all_bboxes = run_object_segmentation_pipeline(
        rgb_data, 
        prompts_list=['bottle', 'cup']
    )
    print("Static data for all objects generated successfully.")

    # --- MODIFIED: Instantiate the new multi-object publishers ---
    mask_publisher = RosInstanceMaskPublisher(topic_name="/pipeline/instance_mask")
    bbox_publisher = RosDetection2DArrayPublisher(topic_name="/pipeline/detections_2d")

    # --- MODIFIED: Create threads with the new list-based arguments ---
    print("Creating and starting continuous publishing threads...")
    # The target methods now expect lists of data
    mask_thread = threading.Thread(target=mask_publisher.publish, args=(all_masks, source_header), daemon=True)
    bbox_thread = threading.Thread(target=bbox_publisher.publish, args=(all_bboxes, source_header), daemon=True)
    
    mask_thread.start()
    bbox_thread.start()

    # --- PRESERVED: Robust shutdown logic using thread.join() ---
    print("\nAll publishers are running. Main thread is now waiting.")
    print("Press Ctrl+C in the terminal to shut down.")
    
    try:
        # The main thread will block here until the threads finish.
        # The threads only finish when rospy.is_shutdown() is true (from Ctrl+C).
        # We only need to join one thread, as rospy.is_shutdown() will stop them all.
        mask_thread.join()
        bbox_thread.join()
        
    except KeyboardInterrupt:
        # This handles the case where Ctrl+C is pressed in the main script's window
        print("\nKeyboardInterrupt received in main thread. Shutting down...")

    finally:
        # This cleanup message will now be printed correctly after Ctrl+C
        print("All threads have finished. Exiting program.")

if __name__ == '__main__':
    run()