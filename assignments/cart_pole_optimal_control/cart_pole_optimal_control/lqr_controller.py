#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import numpy as np
from scipy import linalg
import csv
import os

class CartPoleLQRController(Node):
    def __init__(self):
        super().__init__('cart_pole_lqr_controller')

        log_dir = "/home/rhutvik/ros2_ws/src/cart_pole_optimal_control/cart_pole_optimal_control"
        os.makedirs(log_dir, exist_ok=True)
        self.data_file = open(os.path.join(log_dir, 'lqr_data.csv'), 'w', newline='')
        self.writer = csv.writer(self.data_file)
        self.writer.writerow(['x', 'x_dot', 'theta', 'theta_dot', 'control_force'])

        # System parameters
        self.M = 1.0
        self.m = 0.1
        self.L = 0.5
        self.g = 9.81

        # Linearized system dynamics
        self.A = np.array([
            [0, 1, 0, 0],
            [0, 0, -(self.m * self.g) / self.M, 0],
            [0, 0, 0, 1],
            [0, 0, ((self.M + self.m) * self.g) / (self.M * self.L), 0]
        ])

        self.B = np.array([
            [0],
            [1/self.M],
            [0],
            [-1/(self.M * self.L)]
        ])

        # Fixed Q and R matrices for LQR
        self.Q = np.diag([50.0, 10.0, 100.0, 10.0])
        self.R = np.array([[0.1]])

        # Compute LQR gain
        self.K = self.compute_lqr_gain()
        self.get_logger().info(f'LQR Gain Matrix: {self.K}')

        self.x = np.zeros((4, 1))  # State vector
        self.state_initialized = False

        # Time-related variables
        self.time_step = 0.0
        self.max_time = 50.0  # Total time to stabilize

        # Add initial instability: small perturbation in pole angle and velocity
        self.initial_perturbation()

        # ROS publishers and subscribers
        self.cart_cmd_pub = self.create_publisher(Float64, '/model/cart_pole/joint/cart_to_base/cmd_force', 10)
        self.joint_state_sub = self.create_subscription(
            JointState, '/world/empty/model/cart_pole/joint_state', self.joint_state_callback, 10
        )

        self.timer = self.create_timer(0.005, self.control_loop)

        self.max_force = 20.0  # Saturation limit for control force

        self.get_logger().info('Cart-Pole LQR Controller initialized')

    def compute_lqr_gain(self):
        try:
            P = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
            K = np.linalg.inv(self.R) @ self.B.T @ P
            return K
        except Exception as e:
            self.get_logger().error(f'LQR computation failed: {e}')
            return np.zeros((1, 4))

    def initial_perturbation(self):
        """Introduce a small initial instability (perturb the pole and cart state)"""
        # Slightly perturb the initial conditions to simulate instability
        self.x[0][0] = 0.5  # Initial cart position
        self.x[1][0] = 0.2  # Initial cart velocity
        self.x[2][0] = 0.4  # Initial pole angle (not fully upright)
        self.x[3][0] = 0.3  # Initial pole angular velocity

    def joint_state_callback(self, msg):
        try:
            # Extract cart and pole joint states
            cart_idx = msg.name.index('cart_to_base')
            pole_idx = msg.name.index('pole_joint')

            # Update the state vector x = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
            self.x = np.array([
                [msg.position[cart_idx]],
                [msg.velocity[cart_idx]],
                [msg.position[pole_idx]],
                [msg.velocity[pole_idx]]
            ])

            if not self.state_initialized:
                self.get_logger().info(
                    f'Initial state: cart_pos={self.x[0][0]:.3f}, cart_vel={self.x[1][0]:.3f}, '
                    f'pole_angle={self.x[2][0]:.3f}, pole_vel={self.x[3][0]:.3f}'
                )
                self.state_initialized = True

        except (ValueError, IndexError) as e:
            self.get_logger().warn(f'Joint state processing failed: {e}')

    def control_loop(self):
        if not self.state_initialized:
            return

        # Update the time step
        self.time_step += 0.005
        if self.time_step > self.max_time:
            self.time_step = self.max_time  # Don't exceed the max time

        try:
            # Gradually reduce perturbation over time to simulate instability in the middle
            if self.time_step < self.max_time / 2:
                # Start with higher instability
                self.x[0][0] += 0.01 * np.sin(self.time_step)  # Small oscillation
                self.x[2][0] += 0.01 * np.cos(self.time_step)  # Small oscillation
            else:
                # Gradually reduce instability as time progresses
                self.x[0][0] -= 0.01 * np.sin(self.time_step / 2)  # Reduce oscillation
                self.x[2][0] -= 0.01 * np.cos(self.time_step / 2)  # Reduce oscillation

            # Calculate control input using LQR
            u = -self.K @ self.x
            force = np.clip(float(u[0]), -self.max_force, self.max_force)

            # Log state and control force data to CSV
            self.writer.writerow([self.x[0][0], self.x[1][0], self.x[2][0], self.x[3][0], force])

            # Publish control force to command the cart
            msg = Float64()
            msg.data = force
            self.cart_cmd_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f'Control loop error: {e}')

    def __del__(self):
        try:
            if self.data_file:
                self.data_file.close()
                self.get_logger().info("CSV file closed.")
        except Exception as e:
            self.get_logger().warn(f'Error closing CSV file: {e}')


def main(args=None):
    rclpy.init(args=args)
    controller = CartPoleLQRController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()