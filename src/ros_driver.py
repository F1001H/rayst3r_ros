import logging
import sys
import time
import threading
from typing import List, Dict, Any

import numpy as np
import open3d as o3d
import rospy
import rosgraph.roslogging
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

# === STANDARD MESSAGE IMPORTS (CORRECTED) =====================================
# Import standard messages for detections
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose
# CORRECTED: Import Pose2D from its proper location in geometry_msgs
from geometry_msgs.msg import Pose2D

import hashlib

# Apply concise RospyLogger.findCaller patch for Python 3.11+
print("Applying concise RospyLogger.findCaller patch for Python 3.11+...")
if sys.version_info.major == 3 and sys.version_info.minor >= 11:
    _RospyLogger_class = rosgraph.roslogging.RospyLogger
    if _RospyLogger_class.findCaller is not logging.Logger.findCaller:
        _RospyLogger_class.findCaller = logging.Logger.findCaller
if logging.getLoggerClass() != _RospyLogger_class:
    logging.setLoggerClass(_RospyLogger_class)

def label_to_id(label: str) -> int:
    h = hashlib.sha256(label.encode('utf-8')).digest()
    int_id = int.from_bytes(h[:8], 'big', signed=True)
    return int_id

# ==============================================================================
# === HELPER FUNCTIONS (PRESERVED) =============================================
# ==============================================================================

def o3d_to_ros_pointcloud2(o3d_pc, frame_id="zed2i_left_camera_optical_frame", stamp=None):
    """Converts a single Open3D point cloud object to a ROS PointCloud2 message."""
    if stamp is None:
        stamp = rospy.Time.now() if rospy.core.is_initialized() else rospy.Time.from_sec(time.time())

    header = Header(stamp=stamp, frame_id=frame_id)
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

def numpy_to_ros_image(mask_array, encoding="mono8"):
    """(PRESERVED) Manually converts a 2D NumPy array into a sensor_msgs/Image."""
    if mask_array.dtype != np.uint8:
        mask_array = mask_array.astype(np.uint8)

    img_msg = RosImage()
    img_msg.height, img_msg.width = mask_array.shape
    img_msg.encoding = encoding
    img_msg.step = img_msg.width
    img_msg.is_bigendian = 0
    img_msg.data = mask_array.tobytes()
    return img_msg


# ==============================================================================
# === PUBLISHER CLASSES (CORRECTED) ============================================
# ==============================================================================

class RosInstanceMaskPublisher:
    """Continuously publishes a static set of masks merged into one instance image."""
    def __init__(self, topic_name="/pipeline/instance_mask"):
        self.pub = rospy.Publisher(topic_name, RosImage, queue_size=10)
        rospy.loginfo(f"RosInstanceMaskPublisher: Publishing to '{topic_name}'")

    def publish(self, masks_numpy_list: List[np.ndarray], header: Header):
        """Continuously publishes the instance mask. This is a blocking method."""
        if not masks_numpy_list:
            rospy.logwarn("Cannot publish empty list of masks.")
            return

        height, width = masks_numpy_list[0].shape
        instance_mask = np.zeros((height, width), dtype=np.uint16)
        for i, mask in enumerate(masks_numpy_list):
            instance_mask[mask] = i + 1

        rate = rospy.Rate(1.0)
        while not rospy.is_shutdown():
            try:
                header.stamp = rospy.Time.now()
                img_msg = RosImage(
                    header=header, height=height, width=width, encoding="mono16",
                    is_bigendian=0, step=width * 2, data=instance_mask.tobytes()
                )
                self.pub.publish(img_msg)
                rospy.loginfo_throttle(5, f"Continuously publishing instance mask with {len(masks_numpy_list)} objects...")
                rate.sleep()
            except rospy.ROSInterruptException:
                rospy.loginfo("Mask publishing interrupted.")
                break


class RosDetection2DArrayPublisher:
    """Continuously publishes a static set of 2D detections."""
    def __init__(self, topic_name="/pipeline/detections_2d"):
        self.pub = rospy.Publisher(topic_name, Detection2DArray, queue_size=10)
        rospy.loginfo(f"RosDetection2DArrayPublisher: Publishing to '{topic_name}'")

    def publish(self, bounding_box_dicts: List[Dict[str, Any]], header: Header):
        """
        Publishes detections given dicts with keys:
            - "box": (cx, cy, w, h)
            - "label": str
            - "score": float
        """
        if not bounding_box_dicts:
            rospy.logwarn("Cannot publish empty list of detections.")
            return

        detections = []
        for bb in bounding_box_dicts:
            cx, cy, w, h = bb["box"]
            label = bb["label"]
            score = bb.get("score", 1.0)

            center_pose = Pose2D(x=float(cx), y=float(cy), theta=0.0)
            bbox_msg = BoundingBox2D(center=center_pose, size_x=float(w), size_y=float(h))

            label_id = label_to_id(label)
            hypothesis = ObjectHypothesisWithPose(id=label_id, score=score)

            detections.append(Detection2D(results=[hypothesis], bbox=bbox_msg))

        rate = rospy.Rate(1.0)
        while not rospy.is_shutdown():
            try:
                current_stamp = rospy.Time.now()
                header.stamp = current_stamp
                for det in detections:
                    det.header = header

                array_msg = Detection2DArray(header=header, detections=detections)
                self.pub.publish(array_msg)
                rospy.loginfo_throttle(5, f"Continuously publishing {len(detections)} detections...")
                rate.sleep()
            except rospy.ROSInterruptException:
                rospy.loginfo("Detection publishing interrupted.")
                break

class RosMergedPointCloudPublisher:
    """Continuously publishes a static, merged point cloud."""
    def __init__(self, topic_name="/pipeline/reconstructed_point_cloud"):
        self.pub = rospy.Publisher(topic_name, PointCloud2, queue_size=10)
        rospy.loginfo(f"RosMergedPointCloudPublisher: Publishing to '{topic_name}'")

    def publish(self, o3d_pointclouds: List[o3d.geometry.PointCloud], header: Header):
        """Continuously publishes the merged point cloud. This is a blocking method."""
        if not o3d_pointclouds:
            rospy.logwarn("Cannot publish empty list of point clouds.")
            return

        merged_pc = o3d.geometry.PointCloud()
        for pc in o3d_pointclouds:
            merged_pc += pc

        if not merged_pc.has_points():
            rospy.logwarn("Merged point cloud has no points.")
            return

        rate = rospy.Rate(1.0)
        while not rospy.is_shutdown():
            try:
                ros_msg = o3d_to_ros_pointcloud2(merged_pc, frame_id=header.frame_id, stamp=rospy.Time.now())
                self.pub.publish(ros_msg)
                rospy.loginfo_throttle(5, f"Continuously publishing merged point cloud...")
                rate.sleep()
            except rospy.ROSInterruptException:
                rospy.loginfo("Point cloud publishing interrupted.")
                break

# ==============================================================================
# === SUBSCRIBER CLASS (PRESERVED) =============================================
# ==============================================================================
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

            if depth_image_np is None: return None, None

            if effective_encoding == 'depth_mm':
                rospy.loginfo(f"Received and manually converted 16UC1 Depth image (shape: {depth_image_np.shape})")
                if convert_to_meters:
                    return depth_image_np.astype(np.float32) / 1000.0, ros_depth_msg
                else:
                    return depth_image_np.astype(np.uint16), ros_depth_msg
            elif effective_encoding == 'depth_m':
                rospy.loginfo(f"Received and manually converted 32FC1 Depth image (shape: {depth_image_np.shape})")
                return depth_image_np.astype(np.float32), ros_depth_msg
            else:
                rospy.logerr(f"Expected Depth compatible encoding from Depth topic, but got effective encoding: {effective_encoding}")
                return None, None
                
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
        depth, depth_msg = self.get_depth_image(convert_to_meters=depth_in_meters)
        intrinsics = self.get_camera_intrinsics()

        source_header = depth_msg.header
        print(source_header)
        if rgb is not None and depth is not None and intrinsics is not None:
            rospy.loginfo("Successfully fetched all required image data (manual conversion).")
        else:
            rospy.logwarn("Failed to fetch one or more image data components (manual conversion).")
        return rgb, depth, intrinsics, source_header
# ==============================================================================
# === EXAMPLE USAGE BLOCK WITH THREADING =======================================
# ==============================================================================
if __name__ == '__main__':
    rospy.init_node('ros_data_tools_node', anonymous=True)

    # --- Simulate getting a single set of perception results ---
    rospy.loginfo("Simulating one-time perception for two objects...")
    header = Header(stamp=rospy.Time.now(), frame_id="zed2i_left_camera_optical_frame")

    dummy_masks = [np.zeros((480, 640), dtype=bool) for _ in range(2)]
    dummy_masks[0][100:200, 150:250] = True
    dummy_masks[1][300:350, 400:500] = True

    dummy_bboxes = [(200.0, 150.0, 100.0, 100.0), (450.0, 325.0, 100.0, 50.0)]

    dummy_pcs = [o3d.geometry.PointCloud(), o3d.geometry.PointCloud()]
    points1 = np.random.rand(100, 3) * [0.1, 0.1, 0.05] + [-0.2, 0, 0.5]
    points2 = np.random.rand(80, 3) * [0.05, 0.1, 0.1] + [0.2, 0.1, 0.6]
    dummy_pcs[0].points = o3d.utility.Vector3dVector(points1)
    dummy_pcs[1].points = o3d.utility.Vector3dVector(points2)
    dummy_pcs[0].paint_uniform_color([1, 0, 0]); dummy_pcs[1].paint_uniform_color([0, 1, 0])

    # --- Initialize publishers ---
    rospy.loginfo("Initializing publishers...")
    mask_pub = RosInstanceMaskPublisher()
    bbox_pub = RosDetection2DArrayPublisher()
    pc_pub = RosMergedPointCloudPublisher()

    # --- Setup and start publishing threads ---
    rospy.loginfo("Starting continuous publishing threads...")
    mask_thread = threading.Thread(target=mask_pub.publish, args=(dummy_masks, header), daemon=True)
    bbox_thread = threading.Thread(target=bbox_pub.publish, args=(dummy_bboxes, header), daemon=True)
    pc_thread = threading.Thread(target=pc_pub.publish, args=(dummy_pcs, header), daemon=True)

    mask_thread.start()
    bbox_thread.start()
    pc_thread.start()

    rospy.loginfo("\nAll publishers are running continuously. Press Ctrl+C to stop.")
    rospy.loginfo("You can inspect topics with commands like:")
    rospy.loginfo("rostopic echo /pipeline/instance_mask")
    rospy.loginfo("rostopic echo /pipeline/detections_2d")
    rospy.loginfo("rostopic echo /pipeline/reconstructed_point_cloud")

    rospy.spin()

    rospy.loginfo("Shutting down.")