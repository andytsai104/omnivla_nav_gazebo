"""Episode manager node.

Intended purpose:
- reset the robot / episode state
- randomize start and target goals
- publish prompts and target labels for data collection
- coordinate when an episode starts / ends
"""

import rclpy
from rclpy.node import Node


class EpisodeManagerNode(Node):
    def __init__(self):
        super().__init__("episode_manager_node")
        # TODO:
        # - load prompt templates / goal library
        # - select random goals
        # - reset simulation or reposition robot
        # - publish episode metadata
        self.get_logger().info("Episode manager node started.")


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeManagerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
