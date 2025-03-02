#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """Generate launch description for cylinder landing mission."""
    
    # Get the package share directory
    pkg_share = get_package_share_directory('terrain_mapping_drone_control')
    
    # Get paths
    model_path = os.path.join(pkg_share, 'models')
    
    # Set Gazebo model and resource paths
    gz_model_path = os.path.join(pkg_share, 'models')
    if 'GZ_SIM_MODEL_PATH' in os.environ:
        os.environ['GZ_SIM_MODEL_PATH'] += os.pathsep + gz_model_path
    else:
        os.environ['GZ_SIM_MODEL_PATH'] = gz_model_path

    if 'GZ_SIM_RESOURCE_PATH' in os.environ:
        os.environ['GZ_SIM_RESOURCE_PATH'] += os.pathsep + gz_model_path
    else:
        os.environ['GZ_SIM_RESOURCE_PATH'] = gz_model_path

    # Set initial drone pose - start at 1m height
    os.environ['PX4_GZ_MODEL_POSE'] = '0 0 1 0 0 0'
    
    # Launch PX4 SITL with x500_gimbal
    px4_sitl = ExecuteProcess(
        cmd=['make', 'px4_sitl', 'gz_x500_gimbal'],
        cwd=os.environ['HOME'] + '/PX4-Autopilot',
        output='screen'
    )
    
    # Spawn the cylinder with ArUco marker
    spawn_cylinder = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-file', os.path.join(model_path, 'cylinder', 'model.sdf'),
            '-name', 'cylinder',
            '-x', '5',     # 5 meters in front of the drone
            '-y', '0',     # centered on y-axis
            '-z', '0',     # at ground level
            '-R', '0',     # no roll
            '-P', '0',     # no pitch
            '-Y', '0',     # no yaw
            '-static'      # ensure it's static
        ],
        output='screen'
    )

    # Bridge node for camera and odometry
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='bridge',
        parameters=[{
            'use_sim_time': True,
        }],
        arguments=[
            # Camera topics (one-way from Gazebo to ROS)
            '/camera@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            # PX4 odometry (one-way from Gazebo to ROS)
            '/model/x500_gimbal_0/odometry_with_covariance@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            # Clock (one-way from Gazebo to ROS)
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        remappings=[
            ('/camera', '/drone_camera'),
            ('/camera_info', '/drone_camera_info'),
            ('/model/x500_gimbal_0/odometry_with_covariance', '/fmu/out/vehicle_odometry'),
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulation (Gazebo) clock if true'),
        px4_sitl,
        TimerAction(
            period=2.0,
            actions=[spawn_cylinder]
        ),
        TimerAction(
            period=3.0,
            actions=[bridge]
        )
    ]) 