"""Inference + Nav2 launch file.

Intended purpose:
- start the simulation stack
- start runtime inference node
- start goal resolver / Nav2 bridge
- connect model output to Nav2 goal execution
"""

from launch import LaunchDescription


def generate_launch_description():
    # TODO:
    # 1) Include sim.launch.py
    # 2) Start omnivla_core nodes:
    #    - inference_node
    #    - nav2_goal_bridge_node
    # 3) Load runtime / goal library params
    return LaunchDescription([])
