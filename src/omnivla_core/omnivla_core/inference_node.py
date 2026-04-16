"""Runtime inference node.

Intended purpose:
- subscribe to current RGB image / pose / prompt
- preprocess inputs for OmniVLA / OmniVLA-edge
- run inference
- output either:
  1) semantic goal ID, or
  2) target pose
"""

import rclpy
from rclpy.node import Node


class InferenceNode(Node):
    def __init__(self):
        super().__init__("inference_node")
        # TODO:
        # - declare params for topics / checkpoint / device
        # - create subscriptions for image / pose / prompt
        # - create publisher for predicted goal or pose
        self.get_logger().info("Inference node started.")


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
