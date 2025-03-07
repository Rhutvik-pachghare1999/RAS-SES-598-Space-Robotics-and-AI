#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import numpy as np
import math
from collections import deque
from std_msgs.msg import Float64
from rcl_interfaces.msg import SetParametersResult
from boustrophedon_msgs.msg import PerformanceMetrics


class BoustrophedonController(Node):
    def __init__(self):
        super().__init__('lawnmower_controller')
        self.metrics_publisher = self.create_publisher(PerformanceMetrics, '/performance_metrics', 10)
        self.timer = self.create_timer(0.1, self.publish_metrics)

        # Declare parameters with default values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('Kp_linear', 5.0),
                ('Kd_linear', 0.5),
                ('Kp_angular', 10.0),
                ('Kd_angular', 0.01),
                ('spacing', 1.0)
            ]
        )

        # Get initial parameter values
        self.Kp_linear = self.get_parameter('Kp_linear').value
        self.Kd_linear = self.get_parameter('Kd_linear').value
        self.Kp_angular = self.get_parameter('Kp_angular').value
        self.Kd_angular = self.get_parameter('Kd_angular').value
        self.spacing = self.get_parameter('spacing').value

        # Add parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Create publisher and subscriber
        self.velocity_publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.pose_subscriber = self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)

        # Lawnmower pattern parameters
        self.waypoints = self.generate_waypoints()
        self.current_waypoint = 0

        # Cross-track error calculation
        self.cross_track_errors = deque(maxlen=1000)  # Store last 1000 errors

        # State variables
        self.pose = Pose()
        self.prev_linear_error = 0.0
        self.prev_angular_error = 0.0
        self.prev_time = self.get_clock().now()
        self.current_velocity = Twist()

        # Create control loop timer
        self.timer = self.create_timer(0.1, self.control_loop)

        # Add publisher for cross-track error
        self.error_pub = self.create_publisher(Float64, 'cross_track_error', 10)

        self.get_logger().info('Lawnmower controller started')
        self.get_logger().info(f'Following waypoints: {self.waypoints}')

    def publish_metrics(self):
        if not self.cross_track_errors:
            cross_track_error = 0.0
        else:
            cross_track_error = self.cross_track_errors[-1]  # Use the most recent cross-track error

        avg_cross_track_error = sum(self.cross_track_errors) / len(self.cross_track_errors) if self.cross_track_errors else 0.0

        distance_to_next_waypoint = self.get_distance(
            self.pose.x, self.pose.y,
            self.waypoints[self.current_waypoint][0],
            self.waypoints[self.current_waypoint][1]
        ) if self.current_waypoint < len(self.waypoints) else 0.0

        completion_percentage = (self.current_waypoint / len(self.waypoints)) * 100 if len(self.waypoints) > 0 else 0.0

        msg = PerformanceMetrics()
        msg.cross_track_error = cross_track_error
        msg.average_cross_track_error = avg_cross_track_error
        msg.current_linear_velocity = self.current_velocity.linear.x
        msg.current_angular_velocity = self.current_velocity.angular.z
        msg.distance_to_next_waypoint = distance_to_next_waypoint
        msg.completion_percentage = completion_percentage

        self.metrics_publisher.publish(msg)
        self.get_logger().info(f'Publishing Performance Metrics: {msg}')

    def generate_waypoints(self):
        waypoints = []
        y = 8.0  # Start higher in the window

        while y >= 3.0:  # End at y = 3
            if len(waypoints) % 2 == 0:
                waypoints.append((2.0, y))  # Left to right
                waypoints.append((9.0, y))  # Right edge
            else:
                waypoints.append((9.0, y))  # Right to left
                waypoints.append((2.0, y))  # Left edge

            y -= self.spacing  # Move down by spacing

        return waypoints

    def calculate_cross_track_error(self):
        if self.current_waypoint < 1:
            return 0.0

        start = np.array(self.waypoints[self.current_waypoint - 1])
        end = np.array(self.waypoints[self.current_waypoint])
        pos = np.array([self.pose.x, self.pose.y])

        path_vector = end - start
        path_length = np.linalg.norm(path_vector)
        if path_length < 1e-6:
            return np.linalg.norm(pos - start)

        path_unit = path_vector / path_length
        pos_vector = pos - start
        projection_length = np.dot(pos_vector, path_unit)
        projection_length = max(0, min(path_length, projection_length))
        projected_point = start + projection_length * path_unit
        error_vector = pos - projected_point
        error_sign = np.sign(np.cross(path_unit, error_vector / np.linalg.norm(error_vector)))
        error = np.linalg.norm(error_vector) * error_sign

        self.cross_track_errors.append(abs(error))
        self.error_pub.publish(Float64(data=error))

        return error

    def pose_callback(self, msg):
        self.pose = msg

    def get_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def get_angle(self, x1, y1, x2, y2):
        return math.atan2(y2 - y1, x2 - x1)

    def control_loop(self):
        if self.current_waypoint >= len(self.waypoints):
            self.get_logger().info('Lawnmower pattern complete')
            self.timer.cancel()
            return

        cross_track_error = self.calculate_cross_track_error()
        target_x, target_y = self.waypoints[self.current_waypoint]
        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds / 1e9

        distance = self.get_distance(self.pose.x, self.pose.y, target_x, target_y)
        target_angle = self.get_angle(self.pose.x, self.pose.y, target_x, target_y)
        angular_error = target_angle - self.pose.theta

        while angular_error > math.pi:
            angular_error -= 2 * math.pi
        while angular_error < -math.pi:
            angular_error += 2 * math.pi

        linear_error_derivative = (distance - self.prev_linear_error) / dt
        angular_error_derivative = (angular_error - self.prev_angular_error) / dt

        linear_velocity = self.Kp_linear * distance + self.Kd_linear * linear_error_derivative
        angular_velocity = self.Kp_angular * angular_error + self.Kd_angular * angular_error_derivative

        vel_msg = Twist()
        vel_msg.linear.x = min(linear_velocity, 2.0)
        vel_msg.angular.z = angular_velocity
        self.current_velocity = vel_msg

        self.velocity_publisher.publish(vel_msg)

        self.prev_linear_error = distance
        self.prev_angular_error = angular_error
        self.prev_time = current_time

        if distance < 0.1:
            self.current_waypoint += 1
            self.get_logger().info(f'Reached waypoint {self.current_waypoint}')

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'Kp_linear':
                self.Kp_linear = param.value
            elif param.name == 'Kd_linear':
                self.Kd_linear = param.value
            elif param.name == 'Kp_angular':
                self.Kp_angular = param.value
            elif param.name == 'Kd_angular':
                self.Kd_angular = param.value
            elif param.name == 'spacing':
                self.spacing = param.value
                self.waypoints = self.generate_waypoints()
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    controller = BoustrophedonController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
