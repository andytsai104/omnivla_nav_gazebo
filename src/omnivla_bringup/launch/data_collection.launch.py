"""Data collection mode launch file.

Intended purpose:
- start the simulation stack
- start episode manager + data logger
- load prompt / goal sampling config
- make it easy to collect training samples in one command
"""

from launch import LaunchDescription


def generate_launch_description():
    # TODO:
    # 1) Include sim.launch.py
    # 2) Start omnivla_data nodes:
    #    - episode_manager_node
    #    - data_logger_node
    # 3) Pass config paths for prompts / goals / save directory
    return LaunchDescription([])
