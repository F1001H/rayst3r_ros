import yaml
import os

from reconstruction_engine import run_inference
from annotate_image import run_object_segmentation_pipeline
from ros_driver import RosPointCloudPublisher, RosImageSubscriber

def load_camera_config(config_filename="camera_config.yaml"):
    """Loads camera configuration from a YAML file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_filename)

    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        if not config:
            raise ValueError("Config file is empty or malformed.")
        return config


def run():
    print("Starting reconstruction...")
    camera_config = load_camera_config()
    rgb_topic_name = camera_config.get("rgb_topic")
    depth_topic_name = camera_config.get("depth_topic")
    cam_info_topic_name = camera_config.get("cam_info_topic")
    image_sub = RosImageSubscriber(
        rgb_topic=rgb_topic_name,
        depth_topic=depth_topic_name,
        cam_info_topic=cam_info_topic_name
    )
    rgb_data, depth_data_metric, K_data = image_sub.get_all_data()
    mask= run_object_segmentation_pipeline(rgb_data, ['a pringles can'])
    points = run_inference(rgb_data, depth_data_metric, mask, K_data)
    ros_publisher = RosPointCloudPublisher(default_topic_name="/pipeline/reconstructed_points")
    ros_publisher.publish(points)

if __name__ == '__main__':
    run()