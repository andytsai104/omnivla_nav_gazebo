#!/usr/bin/env python3
"""Data collection mode launch file.

Intended purpose:
- start the simulation stack
- start episode manager + data logger
- load prompt / goal sampling config
- make it easy to collect training samples in one command
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
    bringup_pkg = get_package_share_directory("omnivla_bringup")
    bcr_bot_pkg = get_package_share_directory("bcr_bot")

    use_sim = LaunchConfiguration("use_sim")
    use_rviz = LaunchConfiguration("use_rviz")
    world = LaunchConfiguration("world")
    collection_config = LaunchConfiguration("collection_config")
    goal_library_config = LaunchConfiguration("goal_library_config")

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(bringup_pkg, "launch", "sim.launch.py")
        ),
        launch_arguments={
            "use_rviz": use_rviz,
            "world": world,
        }.items(),
        condition=IfCondition(use_sim),
    )

    episode_manager_node = Node(
        package="omnivla_data",
        executable="episode_manager_node",
        name="episode_manager_node",
        output="screen",
        parameters=[
            collection_config,
            goal_library_config,
        ],
    )

    data_logger_node = Node(
        package="omnivla_data",
        executable="data_logger_node",
        name="data_logger_node",
        output="screen",
        parameters=[
            collection_config,
            goal_library_config,
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim",
            default_value="true",
            description="Whether to launch simulation",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Whether to launch RViz with sim",
        ),
        DeclareLaunchArgument(
            "world",
            default_value=join(bcr_bot_pkg, "worlds", "small_warehouse.sdf"),
            description="Gazebo world file",
        ),
        DeclareLaunchArgument(
            "collection_config",
            default_value=join(bringup_pkg, "config", "collection.yaml"),
            description="Collection config YAML",
        ),
        DeclareLaunchArgument(
            "goal_library_config",
            default_value=join(bringup_pkg, "config", "goal_library.yaml"),
            description="Goal library YAML",
        ),

        sim_launch,
        episode_manager_node,
        data_logger_node,
    ])