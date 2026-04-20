"""Simulation bringup launch file.

Intended purpose:
- launch the BCR Bot in Gazebo Harmonic
- load the warehouse / house world
- start required ROS 2 bridges / robot topics
- optionally start RViz
- prepare the base sim stack before data collection or inference
"""

#!/usr/bin/env python3
"""
Simulation bringup launch file.

Purpose:
- launch the BCR Bot in Gazebo Harmonic
- select a world file
- optionally start RViz
- provide a clean entry point for simulation before data collection or inference
"""

from os.path import join

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    bcr_bot_path = get_package_share_directory("bcr_bot")
    bringup_pkg_path = get_package_share_directory("omnivla_bringup")

    # Launch arguments
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_rviz = LaunchConfiguration("use_rviz")
    world = LaunchConfiguration("world")
    position_x = LaunchConfiguration("position_x")
    position_y = LaunchConfiguration("position_y")
    orientation_yaw = LaunchConfiguration("orientation_yaw")
    camera_enabled = LaunchConfiguration("camera_enabled")
    stereo_camera_enabled = LaunchConfiguration("stereo_camera_enabled")
    two_d_lidar_enabled = LaunchConfiguration("two_d_lidar_enabled")
    odometry_source = LaunchConfiguration("odometry_source")

    default_world = join(bcr_bot_path, "worlds", "small_warehouse.sdf")
    default_rviz_config = join(bringup_pkg_path, "rviz", "sim.rviz")

    # Include official BCR Gazebo launch
    gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(bcr_bot_path, "launch", "gz.launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "world_file": world,
        }.items(),
    )

    # Optional RViz
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", default_rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation clock",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Launch RViz2",
        ),
        DeclareLaunchArgument(
            "world",
            default_value=default_world,
            description="Full path to Gazebo world file",
        ),
        DeclareLaunchArgument(
            "position_x",
            default_value="0.0",
            description="Initial robot x position",
        ),
        DeclareLaunchArgument(
            "position_y",
            default_value="0.0",
            description="Initial robot y position",
        ),
        DeclareLaunchArgument(
            "orientation_yaw",
            default_value="0.0",
            description="Initial robot yaw",
        ),
        DeclareLaunchArgument(
            "camera_enabled",
            default_value="true",
            description="Enable RGB camera",
        ),
        DeclareLaunchArgument(
            "stereo_camera_enabled",
            default_value="true",
            description="Enable stereo camera",
        ),
        DeclareLaunchArgument(
            "two_d_lidar_enabled",
            default_value="true",
            description="Enable 2D lidar",
        ),
        DeclareLaunchArgument(
            "odometry_source",
            default_value="world",
            description="Odometry source for the robot",
        ),

        gz_launch,
        rviz_node,
    ])