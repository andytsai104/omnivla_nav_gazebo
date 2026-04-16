"""Evaluation launch file.

Intended purpose:
- start the full sim + runtime stack
- start automated evaluation runner
- log metrics such as success, time, and collisions
"""

from launch import LaunchDescription


def generate_launch_description():
    # TODO:
    # 1) Include inference_nav.launch.py or sim.launch.py + runtime nodes
    # 2) Start omnivla_eval nodes
    # 3) Load evaluation scenario config
    return LaunchDescription([])
