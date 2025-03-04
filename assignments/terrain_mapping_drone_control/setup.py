from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'terrain_mapping_drone_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('launch/*.launch.py')),
        ('share/' + package_name + '/config',
            glob('config/*')),
        ('share/' + package_name + '/models/terrain',
            glob('models/terrain/*.*')),
        ('share/' + package_name + '/models/terrain/meshes',
            glob('models/terrain/meshes/*.*')),
        ('share/' + package_name + '/models/cylinder',
            glob('models/cylinder/*.*')),
        ('share/' + package_name + '/models/cylinder/materials',
            glob('models/cylinder/materials/*.*')),
        ('lib/' + package_name, [
            'terrain_mapping_drone_control/geometry_tracker.py',
            'terrain_mapping_drone_control/cylinder_landing_node.py',
            'terrain_mapping_drone_control/aruco_tracker.py'
        ])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='A ROS2 package for executing spiral trajectory with PX4',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'terrain_mapping_drone_control_px4_odom_converter = terrain_mapping_drone_control.px4_odom_converter:main',
            'terrain_mapping_drone_control_cylinder_landing_node = terrain_mapping_drone_control.cylinder_landing_node:main',
            'terrain_mapping_drone_control_geometry_tracker = terrain_mapping_drone_control.geometry_tracker:main',
            'terrain_mapping_drone_control_feature_tracker = terrain_mapping_drone_control.feature_tracker:main',
            'terrain_mapping_drone_control_pose_visualizer = terrain_mapping_drone_control.pose_visualizer:main',
            'terrain_mapping_drone_control_spiral_trajectory = terrain_mapping_drone_control.spiral_trajectory:main',
            'terrain_mapping_drone_control_aruco_tracker = terrain_mapping_drone_control.aruco_tracker:main',
        ],
    },
    python_requires='>=3.8'
) 