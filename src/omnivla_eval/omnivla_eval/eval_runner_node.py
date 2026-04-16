"""Evaluation runner node.

Intended purpose:
- run multiple trials automatically
- trigger episodes / prompts / goals
- collect outcome metrics
"""

import rclpy
from rclpy.node import Node


class EvalRunnerNode(Node):
    def __init__(self):
        super().__init__("eval_runner_node")
        # TODO:
        # - load evaluation config
        # - iterate through scenarios / prompts / goals
        # - trigger runtime stack and save results
        self.get_logger().info("Evaluation runner node started.")


def main(args=None):
    rclpy.init(args=args)
    node = EvalRunnerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
