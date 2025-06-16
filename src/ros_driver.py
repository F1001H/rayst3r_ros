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

from sensor_msgs.msg import PointCloud2, PointField, Image as RosImage, CameraInfo


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
        if not rospy.core.is_initialized():
            rospy.init_node('ros_data_tools_node', anonymous=True)
            rospy.loginfo("ROS node initialized by RosImageSubscriber.")        
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

class RosImageSubscriber:
    def __init__(self,
                 rgb_topic="/camera/color/image_raw_placeholder",
                 depth_topic="/camera/depth/image_rect_raw_placeholder",
                 cam_info_topic="/camera/color/camera_info_placeholder",
                 timeout_secs=5.0):
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.cam_info_topic = cam_info_topic
        self.timeout = rospy.Duration(timeout_secs)

        if not rospy.core.is_initialized():
            rospy.init_node('ros_data_tools_node', anonymous=True)
            rospy.loginfo("ROS node initialized by RosImageSubscriber.")

        rospy.loginfo(f"RosImageSubscriber configured for:")
        rospy.loginfo(f"  RGB topic: {self.rgb_topic}")
        rospy.loginfo(f"  Depth topic: {self.depth_topic}")
        rospy.loginfo(f"  CameraInfo topic: {self.cam_info_topic}")

    def _ros_image_to_numpy(self, ros_image_msg):
        """
        Manually converts sensor_msgs/Image data to a NumPy array.
        Handles common encodings like 'rgb8', 'bgr8', 'mono8', '16UC1', '32FC1'.
        Returns (numpy_array, effective_encoding_for_further_processing)
        effective_encoding can be 'rgb', 'mono', 'depth_mm', 'depth_m'
        """
        encoding = ros_image_msg.encoding
        height = ros_image_msg.height
        width = ros_image_msg.width
        step = ros_image_msg.step
        data = ros_image_msg.data
        
        numpy_dtype = None
        channels = 0
        effective_encoding = None

        if encoding in ['rgb8', 'bgr8']:
            numpy_dtype = np.dtype(np.uint8)
            channels = 3
            effective_encoding = 'rgb' 
        elif encoding == 'mono8':
            numpy_dtype = np.dtype(np.uint8)
            channels = 1
            effective_encoding = 'mono'
        elif encoding == '16UC1':
            numpy_dtype = np.dtype(np.uint16)
            channels = 1
            effective_encoding = 'depth_mm'
        elif encoding == '32FC1':
            numpy_dtype = np.dtype(np.float32)
            channels = 1
            effective_encoding = 'depth_m'
        elif encoding == 'rgba8' or encoding == 'bgra8': 
            numpy_dtype = np.dtype(np.uint8)
            channels = 4
            effective_encoding = 'rgba' 
        else:
            rospy.logerr(f"Unsupported image encoding for manual conversion: {encoding}")
            return None, None

        bytes_per_pixel_channel = numpy_dtype.itemsize
        expected_row_size = width * channels * bytes_per_pixel_channel
        
        try:
            if step == expected_row_size:
                image_np = np.frombuffer(data, dtype=numpy_dtype).reshape(height, width, channels) if channels > 1 else \
                           np.frombuffer(data, dtype=numpy_dtype).reshape(height, width)
            else:
                image_np = np.zeros((height, width, channels) if channels > 1 else (height, width), dtype=numpy_dtype)
                for i in range(height):
                    row_data = data[i * step : i * step + expected_row_size]
                    row_np = np.frombuffer(row_data, dtype=numpy_dtype)
                    if channels > 1:
                        image_np[i] = row_np.reshape(width, channels)
                    else:
                        image_np[i] = row_np
        except ValueError as e:
            rospy.logerr(f"Error reshaping image data for encoding {encoding}: {e}. H:{height}, W:{width}, C:{channels}, Step:{step}, ExpectedStep:{expected_row_size}, DataLen:{len(data)}")
            return None, None

        if numpy_dtype.itemsize > 1: 
            dt_ros_endian = numpy_dtype.newbyteorder('>' if ros_image_msg.is_bigendian else '<')
            
            if step == expected_row_size:
                 image_np_correct_endian = np.frombuffer(data, dtype=dt_ros_endian).reshape(image_np.shape)
            else:
                image_np_correct_endian = np.zeros_like(image_np, dtype=dt_ros_endian.base)
                for i in range(height):
                    row_data = data[i * step : i * step + expected_row_size]
                    row_np = np.frombuffer(row_data, dtype=dt_ros_endian)
                    if channels > 1:
                        image_np_correct_endian[i] = row_np.reshape(width, channels)
                    else:
                        image_np_correct_endian[i] = row_np
            
            image_np = image_np_correct_endian.astype(numpy_dtype.base, copy=False)
            rospy.logdebug(f"Applied endianness correction for encoding {encoding} if needed.")


        if encoding == 'bgr8':
            image_np = image_np[:, :, ::-1] 
        elif encoding == 'bgra8': 
            image_np = image_np[:, :, [2,1,0,3]] 

        return image_np, effective_encoding


    def get_rgb_image(self):
        try:
            rospy.loginfo(f"Waiting for RGB message on {self.rgb_topic} (timeout: {self.timeout.to_sec()}s)...")
            ros_image_msg = rospy.wait_for_message(self.rgb_topic, RosImage, timeout=self.timeout)
            
            image_np, effective_encoding = self._ros_image_to_numpy(ros_image_msg)

            if image_np is None: return None

            if effective_encoding == 'rgb':
                rospy.loginfo(f"Received and manually converted RGB image (shape: {image_np.shape})")
                return image_np.astype(np.uint8)
            elif effective_encoding == 'rgba' and image_np.shape[2] == 4:
                rospy.loginfo(f"Received and manually converted RGBA image (shape: {image_np.shape}). Converting to RGB.")
                return image_np[:,:,:3].astype(np.uint8)
            else:
                rospy.logerr(f"Expected RGB compatible encoding from RGB topic, but got effective encoding: {effective_encoding}")
                return None

        except rospy.ROSException as e:
            rospy.logerr(f"Timeout or error waiting for RGB message on {self.rgb_topic}: {e}")
        return None

    def get_depth_image(self, convert_to_meters=True):
        try:
            rospy.loginfo(f"Waiting for Depth message on {self.depth_topic} (timeout: {self.timeout.to_sec()}s)...")
            ros_depth_msg = rospy.wait_for_message(self.depth_topic, RosImage, timeout=self.timeout)
            
            depth_image_np, effective_encoding = self._ros_image_to_numpy(ros_depth_msg)

            if depth_image_np is None: return None

            if effective_encoding == 'depth_mm':
                rospy.loginfo(f"Received and manually converted 16UC1 Depth image (shape: {depth_image_np.shape})")
                if convert_to_meters:
                    return depth_image_np.astype(np.float32) / 1000.0
                else:
                    return depth_image_np.astype(np.uint16)
            elif effective_encoding == 'depth_m':
                rospy.loginfo(f"Received and manually converted 32FC1 Depth image (shape: {depth_image_np.shape})")
                return depth_image_np.astype(np.float32)
            else:
                rospy.logerr(f"Expected Depth compatible encoding from Depth topic, but got effective encoding: {effective_encoding}")
                return None
                
        except rospy.ROSException as e:
            rospy.logerr(f"Timeout or error waiting for Depth message on {self.depth_topic}: {e}")
        return None

    def get_camera_intrinsics(self):
        try:
            rospy.loginfo(f"Waiting for CameraInfo message on {self.cam_info_topic} (timeout: {self.timeout.to_sec()}s)...")
            cam_info_msg = rospy.wait_for_message(self.cam_info_topic, CameraInfo, timeout=self.timeout)
            K_matrix = np.array(cam_info_msg.K, dtype=np.float32).reshape(3, 3)
            rospy.loginfo(f"Received Camera Intrinsics K:\n{K_matrix}")
            return K_matrix
        except rospy.ROSException as e:
            rospy.logerr(f"Timeout or error waiting for CameraInfo message on {self.cam_info_topic}: {e}")
        return None

    def get_all_data(self, depth_in_meters=True):
        rospy.loginfo("Attempting to fetch all RGB, Depth, and CameraInfo data (manual conversion)...")
        rgb = self.get_rgb_image()
        depth = self.get_depth_image(convert_to_meters=depth_in_meters)
        intrinsics = self.get_camera_intrinsics()
        
        if rgb is not None and depth is not None and intrinsics is not None:
            rospy.loginfo("Successfully fetched all required image data (manual conversion).")
        else:
            rospy.logwarn("Failed to fetch one or more image data components (manual conversion).")
        return rgb, depth, intrinsics


if __name__ == '__main__':
    if not rospy.core.is_initialized():
        rospy.init_node('ros_data_tools_node', anonymous=True)
        rospy.loginfo("ROS node initialized in __main__.")

    image_sub = RosImageSubscriber(
        rgb_topic="/zed2i/zed_node/left/image_rect_color",
        depth_topic="/zed2i/zed_node/depth/depth_registered",
        cam_info_topic="/zed2i/zed_node/rgb/camera_info"
    )

    rospy.loginfo("Attempting to get single set of image data...")
    rgb_data, depth_data_metric, K_data = image_sub.get_all_data()

    if rgb_data is not None:
        rospy.loginfo(f"Main: Got RGB image, shape: {rgb_data.shape}")
    else:
        rospy.logwarn("Main: Failed to get RGB image.")

    if depth_data_metric is not None:
        rospy.loginfo(f"Main: Got Depth image (meters), shape: {depth_data_metric.shape}, min: {np.min(depth_data_metric[depth_data_metric>0]) if np.any(depth_data_metric>0) else 'N/A'}, max: {np.max(depth_data_metric)}")
    else:
        rospy.logwarn("Main: Failed to get Depth image.")

    if K_data is not None:
        rospy.loginfo(f"Main: Got Intrinsics K:\n{K_data}")
    else:
        rospy.logwarn("Main: Failed to get Intrinsics.")

    rospy.loginfo("Example finished.")