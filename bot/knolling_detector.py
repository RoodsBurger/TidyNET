#!/usr/bin/env python3
# Main detector module for object recognition in robotics application
import logging
import cv2
import time
import os
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import rclpy  # ROS Client Library for Python
from cv_bridge import CvBridge  # Converts between ROS Image messages and OpenCV images
from sensor_msgs.msg import Image, CameraInfo  # ROS message types
from ultralytics import YOLO  # YOLO object detection model
from interbotix_common_modules.common_robot.robot import create_interbotix_global_node
import tf2_ros  # ROS Transform Library

# Global configuration dictionary for robot and camera settings
CONFIG = {
    'ROBOT_MODEL': 'wx200',  # Robot arm model (WidowX 200)
    'ROBOT_NAME': 'wx200',   # Robot name in ROS namespace
    'ARM_BASE_FRAME': 'wx200/base_link',  # TF frame for robot base
    'CAMERA_FRAME': 'camera_color_optical_frame',  # TF frame for camera
    'ROI_CONFIG_FILE': 'roi_config.yaml',  # File to save/load region of interest
    'YOLO_CONFIDENCE_CAMERA': 0.7,  # Confidence threshold for YOLO detection
    'YOLO_CONFIDENCE_LAYOUT': 0.6,  # Confidence threshold for layout detection
    'CAMERA_MODEL_PATH': './models/multitask.pt',  # Path to YOLO model for camera view
    'LAYOUT_MODEL_PATH': './models/multitask.pt',  # Path to YOLO model for layout view
    'TABLE': {  # Table parameters for plane intersection calculations
        'POINT': [0.360, 0.123, 0.060],  # Reference point on table surface
        'X_MIN': 0.1,  # Minimum x coordinate of workspace
        'X_MAX': 0.4,  # Maximum x coordinate of workspace
        'Z_MIN': 0.03,  # Minimum height of workspace
        'Z_MAX': 0.06   # Maximum height of workspace
    }
}

@dataclass(frozen=False, eq=True)
class DetectedObject:
    """
    Dataclass representing a detected object with its properties
    
    Attributes:
        object_position: 3D position (x, y, z) in robot base frame
        bbox: Bounding box information including dimensions and corner points
        class_name: Object class name from YOLO model
        confidence: Detection confidence score
        angle: Orientation angle in radians
        avg_color: Average RGB color of the object
        dimensions: Physical dimensions (width, height) in meters
    """
    object_position: Tuple[float, float, float]
    bbox: Optional[Dict[str, Any]] = None
    class_name: str = ""
    confidence: float = 0.0
    angle: float = 0.0
    avg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dimensions: Tuple[float, float] = (0.0, 0.0)
    
    def __hash__(self):
        """Custom hash implementation for using objects in sets/dicts"""
        return hash((
            self.object_position,
            self.class_name,
            self.confidence,
            self.angle,
            self.avg_color,
            self.dimensions
        ))
    
    def __eq__(self, other):
        """Custom equality check for comparing detected objects"""
        if not isinstance(other, DetectedObject):
            return False
        return (
            self.object_position == other.object_position and
            self.class_name == other.class_name and
            self.confidence == other.confidence and
            self.angle == other.angle and
            self.avg_color == other.avg_color and
            self.dimensions == other.dimensions
        )

class YOLODetector:
    """
    Class for detecting objects using YOLO model and converting between
    image coordinates and robot-frame coordinates
    """
    def __init__(self, camera_model_path=None, layout_model_path=None):
        """
        Initialize the detector with YOLO models and ROS components
        
        Args:
            camera_model_path: Optional custom path to YOLO model for camera view
            layout_model_path: Optional custom path to YOLO model for layout view
        """
        # Set model paths, using CONFIG defaults if not provided
        self.camera_model_path = camera_model_path or CONFIG['CAMERA_MODEL_PATH']
        self.layout_model_path = layout_model_path or CONFIG['LAYOUT_MODEL_PATH']
        
        # Load YOLO models
        self.camera_model = YOLO(self.camera_model_path)
        self.camera_model.verbose = False
        
        # Use separate layout model only if path is different
        if self.layout_model_path != self.camera_model_path:
            self.layout_model = YOLO(self.layout_model_path)
            self.layout_model.verbose = False
        else:
            self.layout_model = self.camera_model
            
        # Initialize OpenCV bridge for ROS image conversion
        self.bridge = CvBridge()
        
        # Initialize camera data placeholders
        self.latest_color_image = None  # Latest camera frame
        self.camera_matrix = None       # Camera intrinsic matrix
        self.roi = None                 # Region of interest in image
        self.detected_objects: List[DetectedObject] = []  # Detected objects list
        
        # Initialize ROS node and subscribers
        self.global_node = create_interbotix_global_node()
        
        # Subscribe to camera image topic
        self.color_sub = self.global_node.create_subscription(
            Image, '/camera/camera/color/image_raw', self._color_callback, 10)
            
        # Subscribe to camera calibration info topic
        self.info_sub = self.global_node.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self._info_callback, 10)
            
        # Initialize transform listener for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.global_node)
        
        # Load saved ROI if available
        self.load_roi()

    def _color_callback(self, msg):
        """
        ROS callback for color image messages
        
        Args:
            msg: ROS Image message
        """
        try:
            # Convert ROS Image message to OpenCV image
            self.latest_color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            logging.error(f"Error converting color image: {e}")

    def _info_callback(self, msg):
        """
        ROS callback for camera info messages
        
        Args:
            msg: ROS CameraInfo message
        """
        if self.camera_matrix is None:
            try:
                # Extract camera intrinsic matrix from message
                self.camera_matrix = np.array(msg.k).reshape(3, 3)
                logging.info("Camera matrix set")
            except Exception as e:
                logging.error(f"Error parsing camera info: {e}")

    def tensor_to_numpy(self, tensor):
        """
        Convert PyTorch tensor to NumPy array safely
        
        Args:
            tensor: PyTorch tensor or tensor-like object
            
        Returns:
            NumPy array version of the tensor
        """
        if hasattr(tensor, 'cpu') and callable(getattr(tensor, 'cpu')):
            tensor = tensor.cpu()
        if hasattr(tensor, 'numpy') and callable(getattr(tensor, 'numpy')):
            return tensor.numpy()
        return tensor

    def load_roi(self):
        """Load region of interest from YAML file if it exists"""
        if os.path.exists(CONFIG['ROI_CONFIG_FILE']):
            try:
                import yaml
                with open(CONFIG['ROI_CONFIG_FILE'], 'r') as f:
                    config = yaml.safe_load(f)
                    roi_loaded = config.get('roi')
                    if roi_loaded is not None:
                        self.roi = tuple(roi_loaded)
            except: pass

    def save_roi(self, roi):
        """
        Save region of interest to YAML file
        
        Args:
            roi: ROI tuple (x, y, width, height)
        """
        try:
            import yaml
            with open(CONFIG['ROI_CONFIG_FILE'], 'w') as f:
                yaml.dump({'roi': list(roi)}, f)
        except: pass

    def wait_for_image(self, timeout=5.0) -> bool:
        """
        Wait until camera image and calibration are available
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if image and camera matrix are available, False if timeout
        """
        start = time.time()
        while self.latest_color_image is None or self.camera_matrix is None:
            if time.time() - start > timeout:
                logging.error("Timeout waiting for image or camera info")
                return False
            rclpy.spin_once(self.global_node, timeout_sec=0.1)
        return True

    def select_roi(self) -> bool:
        """
        Open window for user to manually select region of interest
        
        Returns:
            True if valid ROI selected, False otherwise
        """
        if not self.wait_for_image(): return False
        cv2.namedWindow('Select ROI')
        self.roi = cv2.selectROI('Select ROI', self.latest_color_image, False)
        cv2.destroyWindow('Select ROI')
        valid = bool(self.roi[2] > 0 and self.roi[3] > 0)
        if valid: self.save_roi(self.roi)
        return valid
    
    def get_transform(self, target_frame, source_frame, timeout=2.0):
        """
        Get transformation between two coordinate frames
        
        Args:
            target_frame: Target coordinate frame name
            source_frame: Source coordinate frame name
            timeout: Maximum time to wait for transform
            
        Returns:
            Transform object or None if not available
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                return self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            except:
                rclpy.spin_once(self.global_node, timeout_sec=0.1)
        return None

    def get_camera_ray(self, pixel_x, pixel_y):
        """
        Convert image pixel to camera ray direction
        
        Args:
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image
            
        Returns:
            Normalized ray direction vector in camera frame
        """
        if self.camera_matrix is None: return None
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        # Calculate ray using pinhole camera model
        ray = np.array([(pixel_x - cx) / fx, (pixel_y - cy) / fy, 1.0])
        return ray / np.linalg.norm(ray)  # Normalize to unit vector

    def get_camera_transform(self):
        """
        Get camera position and orientation in robot base frame
        
        Returns:
            Tuple of (camera_position, rotation_matrix) or (None, None) if not available
        """
        transform = self.get_transform(CONFIG['CAMERA_FRAME'], CONFIG['ARM_BASE_FRAME'])
        if transform is None: return None, None
        
        # Extract translation component
        trans = transform.transform.translation
        t = np.array([trans.x, trans.y, trans.z])
        
        # Extract rotation component (quaternion to matrix)
        rot = transform.transform.rotation
        qx, qy, qz, qw = rot.x, rot.y, rot.z, rot.w
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])
        
        # Ensure rotation matrix is orthogonal using SVD
        u, s, vh = np.linalg.svd(R)
        R = u @ vh
        
        # Return camera position and rotation matrix
        return -R.T @ t, R

    def image_to_z(self, pixel_x, pixel_y) -> float:
        """
        Calculate Z coordinate of table intersection point for a given pixel
        
        Args:
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image
            
        Returns:
            Z coordinate in robot base frame or None if calculation fails
        """
        # Get ray in camera frame
        ray_cam = self.get_camera_ray(pixel_x, pixel_y)
        if ray_cam is None: return None
        
        # Get camera transform
        camera_pos, R = self.get_camera_transform()
        if camera_pos is None: return None
        
        # Transform ray to robot base frame
        ray_base = R.T @ ray_cam
        
        # Set up plane intersection calculation
        table_point = np.array(CONFIG['TABLE']['POINT'])
        table_normal = R[:, 2]  # Z-axis of camera is normal to table
        
        # Calculate ray-plane intersection
        denom = np.dot(table_normal, ray_base)
        if np.abs(denom) < 1e-6: return None  # Ray is parallel to plane
        
        # Calculate plane equation: dot(normal, point) + D = 0
        D = -np.dot(table_normal, table_point)
        
        # Calculate intersection parameter
        lambda_param = -(np.dot(table_normal, camera_pos) + D) / denom
        
        # Calculate intersection point
        intersection = camera_pos + lambda_param * ray_base
        
        # Map X coordinate to normalized range and interpolate Z value
        normalized_x = (intersection[0] - CONFIG['TABLE']['X_MIN']) / (CONFIG['TABLE']['X_MAX'] - CONFIG['TABLE']['X_MIN'])
        return CONFIG['TABLE']['Z_MIN'] + normalized_x * (CONFIG['TABLE']['Z_MAX'] - CONFIG['TABLE']['Z_MIN'])

    def image_to_xy(self, pixel_x, pixel_y, plane_z=0.0) -> Optional[Tuple[float, float]]:
        """
        Convert image pixel to XY coordinates in robot base frame
        
        Args:
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image
            plane_z: Z coordinate of the plane to intersect with
            
        Returns:
            Tuple of (X, Y) coordinates in robot base frame or None if calculation fails
        """
        # Get ray in camera frame
        ray_cam = self.get_camera_ray(pixel_x, pixel_y)
        if ray_cam is None: return None
        
        # Get camera transform
        camera_pos, R = self.get_camera_transform()
        if camera_pos is None: return None
        
        # Transform ray to robot base frame
        ray_base = R.T @ ray_cam
        
        # Check if ray is parallel to XY plane
        if abs(ray_base[2]) < 1e-6: return None
        
        # Calculate intersection parameter for Z=plane_z plane
        lambda_param = (plane_z - camera_pos[2]) / ray_base[2]
        
        # Calculate intersection point
        intersection = camera_pos + lambda_param * ray_base
        
        # Return XY coordinates
        return (intersection[0], intersection[1])
    
    def detect_objects(self) -> bool:
        """
        Detect objects in the latest camera image
        
        Returns:
            True if any objects were detected, False otherwise
        """
        if self.latest_color_image is None: return False
        
        # Crop image to ROI if defined
        if self.roi:
            x, y, w, h = self.roi
            detect_image = self.latest_color_image[y:y+h, x:x+w]
            roi_offset = (x, y)
        else:
            detect_image = self.latest_color_image
            roi_offset = (0, 0)

        # Run YOLO model prediction
        results = self.camera_model.predict(detect_image, conf=CONFIG['YOLO_CONFIDENCE_CAMERA'], verbose=False)
        if not results or len(results) == 0: return False
        
        # Clear previous detections
        self.detected_objects.clear()
        
        # Process each detection result
        for result in results:
            try:
                # Extract oriented bounding box data
                obb_data = self.tensor_to_numpy(result.obb.xywhr) if hasattr(result.obb, 'xywhr') else None
                cls_data = self.tensor_to_numpy(result.obb.cls) if hasattr(result.obb, 'cls') else None
                conf_data = self.tensor_to_numpy(result.obb.conf) if hasattr(result.obb, 'conf') else None
                poly_data = self.tensor_to_numpy(result.obb.xyxyxyxy) if hasattr(result.obb, 'xyxyxyxy') else None
                
                # Skip if any required data is missing
                if obb_data is None or cls_data is None or conf_data is None:
                    continue
                
                # Process each detected object
                for i in range(len(obb_data)):
                    if i >= len(cls_data) or i >= len(conf_data):
                        continue
                    
                    # Extract basic detection data
                    xywhr = obb_data[i]
                    class_id = int(cls_data[i]) if isinstance(cls_data[i], (int, float)) else int(cls_data[i].item())
                    confidence = float(conf_data[i]) if isinstance(conf_data[i], (int, float)) else float(conf_data[i].item())
                    
                    # Skip if not enough information
                    if len(xywhr) < 5:
                        continue
                    
                    # Extract bounding box parameters
                    cx, cy, w, h, angle = xywhr[:5]
                    
                    # Adjust coordinates if using ROI
                    if self.roi:
                        cx += roi_offset[0]
                        cy += roi_offset[1]
                    
                    # Calculate 3D position
                    z = self.image_to_z(cx, cy)
                    if z is None: continue
                    
                    world_xy = self.image_to_xy(cx, cy, z)
                    if world_xy is None: continue
                    
                    world_coords = world_xy + (z,)
                    
                    # Calculate physical dimensions
                    world_w, world_h = 0.0, 0.0
                    if w > 0 and h > 0:
                        # Calculate corner points in world coordinates
                        corner1 = self.image_to_xy(cx - w/2, cy, z)
                        corner2 = self.image_to_xy(cx + w/2, cy, z)
                        corner3 = self.image_to_xy(cx, cy - h/2, z)
                        corner4 = self.image_to_xy(cx, cy + h/2, z)
                        
                        # Calculate physical width and height
                        if corner1 and corner2:
                            world_w = np.linalg.norm(np.array(corner2) - np.array(corner1))
                        if corner3 and corner4:
                            world_h = np.linalg.norm(np.array(corner4) - np.array(corner3))
                    
                    # Extract polygon vertices if available
                    polygon_points = []
                    if poly_data is not None and i < len(poly_data):
                        img_polygon = poly_data[i].reshape(-1, 2)
                        for point in img_polygon:
                            px, py = point
                            if self.roi:
                                px += roi_offset[0]
                                py += roi_offset[1]
                            world_point = self.image_to_xy(px, py, z)
                            if world_point:
                                polygon_points.append(world_point)
                    
                    # Calculate ROI for average color extraction
                    x1, y1 = int(cx - w/2), int(cy - h/2)
                    x2, y2 = int(cx + w/2), int(cy + h/2)
                    
                    if self.roi:
                        x1 += roi_offset[0]
                        y1 += roi_offset[1]
                        x2 += roi_offset[0]
                        y2 += roi_offset[1]
                    
                    # Calculate average color
                    avg_color = (0, 0, 0)
                    if x1 < x2 and y1 < y2:
                        try:
                            roi_crop = self.latest_color_image[y1:y2, x1:x2]
                            avg_color = cv2.mean(roi_crop)[:3]
                        except: pass
                    
                    # Create bounding box dictionary
                    bbox = {
                        'width': world_w,
                        'height': world_h,
                        'points': polygon_points
                    }

                    print(f'BOX: {bbox}, {world_w}, {world_h}')
                    
                    # Create and add DetectedObject
                    self.detected_objects.append(DetectedObject(
                        object_position=world_coords,
                        class_name=self.camera_model.names[class_id],
                        confidence=confidence,
                        angle=angle,
                        avg_color=avg_color,
                        bbox=bbox,
                        dimensions=(world_w, world_h)
                    ))
                    logging.info(f"Detected {self.camera_model.names[class_id]} at {world_coords} with angle {angle:.4f}")
            except Exception as e:
                logging.error(f"Error with detection: {e}")
                
        return len(self.detected_objects) > 0

# Main execution if script is run directly
if __name__ == '__main__':
    import sys
    rclpy.init()  # Initialize ROS client library
    detector = YOLODetector()  # Create detector instance
    if not detector.wait_for_image(): sys.exit(1)  # Wait for camera image
    detector.select_roi()  # Let user select region of interest
    if not detector.detect_objects(): sys.exit(1)  # Run object detection
    rclpy.shutdown()  # Shutdown ROS client library