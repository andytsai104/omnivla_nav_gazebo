"""Data logger node.

Intended purpose:
- subscribe to image / pose / prompt / goal topics
- save synchronized samples for training
- write metadata to disk in a format that is easy to export later
"""

import rclpy
from rclpy.node import Node


class DataLoggerNode(Node):
    def __init__(self):
        super().__init__("data_logger_node")
        # TODO:
        # - declare output path params
        # - subscribe to required topics
        # - save image + json/csv metadata per sample or per episode
        self.get_logger().info("Data logger node started.")


def main(args=None):
    rclpy.init(args=args)
    node = DataLoggerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
