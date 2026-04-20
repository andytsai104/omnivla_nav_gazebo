"""Evaluation launch file.

Intended purpose:
- start the full sim + runtime stack
- start automated evaluation runner
- log metrics such as success, time, and collisions
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
    eval_config = LaunchConfiguration("eval_config")

    inference_nav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(bringup_pkg, "launch", "inference_nav.launch.py")
        ),
        launch_arguments={
            "use_rviz": use_rviz,
            "world": world,
            "runtime_config": runtime_config,
            "goal_library_config": goal_library_config,
        }.items(),
    )

    eval_runner_node = Node(
        package="omnivla_eval",
        executable="eval_runner_node",
        name="eval_runner_node",
        output="screen",
        parameters=[
            eval_config,
            {"goal_library_path": goal_library_config},
        ],
    )

    result_logger_node = Node(
        package="omnivla_eval",
        executable="result_logger_node",
        name="result_logger_node",
        output="screen",
        parameters=[
            eval_config,
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_rviz",
            default_value="false",
            description="Whether to launch RViz during evaluation",
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
        DeclareLaunchArgument(
            "eval_config",
            default_value=join(bringup_pkg, "config", "eval.yaml"),
            description="Evaluation config YAML",
        ),

        inference_nav_launch,
        eval_runner_node,
        result_logger_node,
    ])