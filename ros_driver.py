import logging


print("Applying concise RospyLogger.findCaller patch for Python 3.11+...")
import rosgraph.roslogging
_RospyLogger_class = rosgraph.roslogging.RospyLogger
_RospyLogger_class.findCaller = logging.Logger.findCaller

if logging.getLoggerClass() != _RospyLogger_class:
    logging.setLoggerClass(_RospyLogger_class)


import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import open3d as o3d
import numpy as np
import time


def o3d_to_ros_pointcloud2(o3d_pc, frame_id="map", stamp=None):
    if stamp is None:
        stamp = rospy.Time.now() if rospy.core.is_initialized() else rospy.Time.from_sec(time.time())

    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id

    points_xyz = np.asarray(o3d_pc.points, dtype=np.float32)
    n_points = points_xyz.shape[0]

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 12
    cloud_data_list = [points_xyz]

    if o3d_pc.has_colors():
        colors_rgb = np.asarray(o3d_pc.colors, dtype=np.float32)
        colors_rgb_u8 = (colors_rgb * 255).astype(np.uint8)
        packed_bgr_buffer = np.zeros(n_points, dtype=np.uint32)
        packed_bgr_buffer |= colors_rgb_u8[:, 2]
        packed_bgr_buffer |= np.uint32(colors_rgb_u8[:, 1]) << 8
        packed_bgr_buffer |= np.uint32(colors_rgb_u8[:, 0]) << 16
        packed_bgr_float32 = packed_bgr_buffer.copy().view(np.float32)
        fields.append(PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1))
        point_step += 4
        cloud_data_list.append(packed_bgr_float32.reshape(-1,1))

    combined_data = np.hstack(cloud_data_list).astype(np.float32) if len(cloud_data_list) > 1 else points_xyz.astype(np.float32)
    cloud_data_bytes = combined_data.tobytes()

    return PointCloud2(
        header=header, height=1, width=n_points, is_dense=True, is_bigendian=False,
        fields=fields, point_step=point_step, row_step=point_step * n_points, data=cloud_data_bytes
    )

class RosPointCloudPublisher:
    def __init__(self, default_topic_name="/my_o3d_cloud", default_frame_id="world"):
        self.pub = None
        self.topic_name = default_topic_name
        self.frame_id = default_frame_id
        self.is_setup = False
        print('here')
        self.setup_publisher()

    def setup_publisher(self):
        rospy.init_node('pointcloud_publisher')
        self.topic_name = rospy.get_param("~pointcloud_topic", self.topic_name)
        self.frame_id = rospy.get_param("~pointcloud_frame_id", self.frame_id)
        self.pub = rospy.Publisher(self.topic_name, PointCloud2, queue_size=5)
        self.is_setup = True
        print(f"RosPointCloudPublisher: Publishing to '{self.topic_name}'")
        return True

    def publish(self, o3d_pointcloud, stamp=None):
        current_stamp = stamp if stamp is not None else (rospy.Time.now() if rospy.core.is_initialized() else None)
        ros_msg = o3d_to_ros_pointcloud2(o3d_pointcloud, frame_id=self.frame_id, stamp=current_stamp)
        rate = rospy.Rate(1.0)
        self.pub.publish(ros_msg)
        while not rospy.is_shutdown():
            current_stamp = stamp
            if current_stamp is None:
                 current_stamp = rospy.Time.now() if rospy.core.is_initialized() else rospy.Time.from_sec(time.time())

            ros_msg = o3d_to_ros_pointcloud2(o3d_pointcloud, frame_id=self.frame_id, stamp=current_stamp)
            self.pub.publish(ros_msg)
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                rospy.loginfo(f"Continuous publishing to '{self.topic_name}' interrupted.")
                break 


if __name__ == '__main__':
    pub = RosPointCloudPublisher()