#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleCommand, VehicleControlMode, OffboardControlMode, TrajectorySetpoint, VehicleOdometry
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped
from aruco_msgs.msg import Marker, MarkerArray
import tf2_ros
import math
import time

class CylinderLandingNode(Node):
    """
    ROS2 Node for autonomous drone takeoff, cylinder detection, and precision landing.
    
    Assignment Tasks:
    1. Implement autonomous takeoff to a safe height
    2. Detect the ArUco marker on top of the cylinder using computer vision
    3. Navigate to and perform precision landing on top of the cylinder
    
    The cylinder is:
    - 10 meters tall
    - 1 meter in diameter
    - Has an ArUco marker (ID 0) on top (0.8m x 0.8m)
    - Located 5 meters in front of the starting position
    """

    def __init__(self):
        super().__init__('cylinder_landing_node')

        # Initialize CV bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Initialize tf2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Create publishers
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10)
        
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        
        # Create subscribers
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, 10)
        
        self.camera_subscriber = self.create_subscription(
            Image, '/drone_camera', self.camera_callback, 10)
        
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo, '/drone_camera_info', self.camera_info_callback, 10)
            
        self.marker_subscriber = self.create_subscription(
            MarkerArray, '/aruco_marker_publisher/markers', self.marker_callback, 10)

        # Initialize state variables
        self.current_position = None
        self.current_orientation = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.marker_position = None
        self.flight_state = 'INIT'
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Timer for publishing control messages
        self.create_timer(0.1, self.control_loop)

    def camera_info_callback(self, msg):
        """Store camera calibration data for ArUco marker detection."""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)

    def marker_callback(self, msg):
        """Process detected ArUco markers."""
        for marker in msg.markers:
            if marker.id == 0:  # We're looking for marker ID 0
                self.marker_position = marker.pose.pose
                # Broadcast the marker transform
                transform = TransformStamped()
                transform.header = marker.header
                transform.child_frame_id = f'aruco_marker_{marker.id}'
                transform.transform.translation.x = marker.pose.pose.position.x
                transform.transform.translation.y = marker.pose.pose.position.y
                transform.transform.translation.z = marker.pose.pose.position.z
                transform.transform.rotation = marker.pose.pose.orientation
                self.tf_broadcaster.sendTransform(transform)

    def camera_callback(self, msg):
        """Process camera image for ArUco marker detection."""
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect ArUco markers
            corners, ids, rejected = self.detector.detectMarkers(gray)
            
            if ids is not None and 0 in ids:
                # Draw detected markers
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                
                # Get the index of marker ID 0
                marker_idx = np.where(ids == 0)[0][0]
                
                # Estimate pose of marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[marker_idx]], 0.8,  # marker size is 0.8 meters
                    self.camera_matrix, self.dist_coeffs)
                
                # Draw axis for the marker
                cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs,
                                rvecs[0], tvecs[0], 0.4)
                
                # Convert to ROS pose and publish transform
                transform = TransformStamped()
                transform.header.stamp = self.get_clock().now().to_msg()
                transform.header.frame_id = msg.header.frame_id
                transform.child_frame_id = 'aruco_marker_0'
                
                # Set translation
                transform.transform.translation.x = tvecs[0][0][0]
                transform.transform.translation.y = tvecs[0][0][1]
                transform.transform.translation.z = tvecs[0][0][2]
                
                # Convert rotation vector to quaternion
                rot_matrix, _ = cv2.Rodrigues(rvecs[0])
                quat = self.rotation_matrix_to_quaternion(rot_matrix)
                transform.transform.rotation.x = quat[0]
                transform.transform.rotation.y = quat[1]
                transform.transform.rotation.z = quat[2]
                transform.transform.rotation.w = quat[3]
                
                # Broadcast the transform
                self.tf_broadcaster.sendTransform(transform)
            
            # Convert back to ROS Image and publish for visualization
            debug_msg = self.cv_bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            debug_msg.header = msg.header
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion."""
        trace = np.trace(R)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        return [qx, qy, qz, qw]

    def vehicle_odometry_callback(self, msg):
        """Store vehicle position and orientation from odometry."""
        self.current_position = [msg.position[0], msg.position[1], msg.position[2]]
        self.current_orientation = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]

    def publish_offboard_control_mode(self):
        """
        TODO: Publish offboard control mode message.
        Set position, velocity, acceleration, attitude, and rates as needed.
        """
        pass

    def publish_trajectory_setpoint(self, x, y, z, yaw):
        """
        TODO: Publish trajectory setpoint for position control.
        """
        pass

    def arm(self):
        """
        TODO: Send arm command to vehicle.
        """
        pass

    def takeoff(self):
        """
        TODO: Implement autonomous takeoff sequence.
        
        Steps:
        1. Switch to offboard mode
        2. Arm the vehicle
        3. Command takeoff to safe height (e.g., 5 meters)
        4. Verify takeoff completion
        """
        pass

    def detect_and_navigate(self):
        """
        TODO: Implement marker detection and navigation logic.
        
        Steps:
        1. Search for ArUco marker if not detected
        2. Once detected, calculate relative position
        3. Plan trajectory to position above marker
        4. Maintain position while preparing for landing
        """
        pass

    def precision_landing(self):
        """
        TODO: Implement precision landing sequence.
        
        Steps:
        1. Verify stable position above marker
        2. Begin controlled descent
        3. Monitor marker visibility during descent
        4. Perform final landing sequence
        """
        pass

    def control_loop(self):
        """
        TODO: Implement main control loop state machine.
        
        States:
        - INIT: Initial state
        - TAKEOFF: Performing takeoff sequence
        - SEARCH: Searching for marker
        - APPROACH: Moving to landing position
        - LANDING: Executing landing sequence
        - LANDED: Mission complete
        """
        if self.flight_state == 'INIT':
            # Initialize flight
            pass
        elif self.flight_state == 'TAKEOFF':
            # Execute takeoff sequence
            pass
        elif self.flight_state == 'SEARCH':
            # Search for marker
            pass
        elif self.flight_state == 'APPROACH':
            # Approach marker
            pass
        elif self.flight_state == 'LANDING':
            # Execute landing
            pass
        elif self.flight_state == 'LANDED':
            # Mission complete
            pass

def main(args=None):
    rclpy.init(args=args)
    node = CylinderLandingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 