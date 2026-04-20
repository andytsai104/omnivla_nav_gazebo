"""Inference + Nav2 launch file.

Intended purpose:
- start the simulation stack
- start runtime inference node
- start goal resolver / Nav2 bridge
- connect model output to Nav2 goal execution
"""

from os.path import join

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    bringup_pkg = get_package_share_directory("omnivla_bringup")
    bcr_bot_pkg = get_package_share_directory("bcr_bot")

    use_rviz = LaunchConfiguration("use_rviz")
    world = LaunchConfiguration("world")
    runtime_config = LaunchConfiguration("runtime_config")
    goal_library_config = LaunchConfiguration("goal_library_config")

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(bringup_pkg, "launch", "sim.launch.py")
        ),
        launch_arguments={
            "use_rviz": use_rviz,
            "world": world,
        }.items(),
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(bcr_bot_pkg, "launch", "nav2.launch.py")
        ),
        launch_arguments={
            "use_sim_time": "true",
            "autostart": "true",
        }.items(),
    )

    inference_node = Node(
        package="omnivla_core",
        executable="inference_node",
        name="inference_node",
        output="screen",
        parameters=[
            runtime_config,
            {"goal_library_path": goal_library_config},
        ],
    )

    nav2_goal_bridge_node = Node(
        package="omnivla_core",
        executable="nav2_goal_bridge_node",
        name="nav2_goal_bridge_node",
        output="screen",
        parameters=[
            runtime_config,
            {"goal_library_path": goal_library_config},
        ],
    )

    basefootprint_to_baselink_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="basefootprint_to_baselink",
        arguments=["0", "0", "0", "0", "0", "0", "base_footprint", "base_link"],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Whether to launch RViz",
        ),
        DeclareLaunchArgument(
            "world",
            default_value=join(bcr_bot_pkg, "worlds", "small_warehouse.sdf"),
            description="Gazebo world file",
        ),
        DeclareLaunchArgument(
            "runtime_config",
            default_value=join(bringup_pkg, "config", "runtime.yaml"),
            description="Runtime config YAML",
        ),
        DeclareLaunchArgument(
            "goal_library_config",
            default_value=join(bringup_pkg, "config", "goal_library.yaml"),
            description="Goal library YAML",
        ),

        sim_launch,
        basefootprint_to_baselink_node,
        nav2_launch,
        inference_node,
        nav2_goal_bridge_node,
    ])