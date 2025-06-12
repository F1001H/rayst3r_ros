import sys
import os
import time 

from reconstruction_engine import run_inference
import open3d as o3d
import numpy as np

DATA_DIRECTORY = "/home/fabian/3d_reconstruction/src/zed_output/"

def run():
    print("Starting reconstruction...")
    points = run_inference(DATA_DIRECTORY)
    from ros_driver import RosPointCloudPublisher
    ros_publisher = RosPointCloudPublisher(default_topic_name="/pipeline/reconstructed_points")
    ros_publisher.publish(points)

if __name__ == '__main__':
    run()