"""Simulation bringup launch file.

Intended purpose:
- launch the BCR Bot in Gazebo Harmonic
- load the warehouse / house world
- start required ROS 2 bridges / robot topics
- optionally start RViz
- prepare the base sim stack before data collection or inference
"""

from launch import LaunchDescription


def generate_launch_description():
    # TODO:
    # 1) Include bcr_bot Gazebo launch file
    # 2) Select world file / map / rviz config
    # 3) Add any bridge or robot_state_publisher nodes if needed
    # 4) Expose launch arguments such as:
    #    - use_rviz
    #    - world
    #    - use_sim_time
    return LaunchDescription([])
