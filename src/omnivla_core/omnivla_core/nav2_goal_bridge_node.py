"""Nav2 goal bridge node.

Intended purpose:
- receive semantic goal ID or predicted pose from the model
- convert that output into a Nav2-compatible goal
- send the goal to Nav2 for execution
"""

import rclpy
from rclpy.node import Node


class Nav2GoalBridgeNode(Node):
    def __init__(self):
        super().__init__("nav2_goal_bridge_node")
        # TODO:
        # - subscribe to predicted goal output
        # - resolve goal ID using goal_library if needed
        # - publish / send PoseStamped to Nav2
        self.get_logger().info("Nav2 goal bridge node started.")


def main(args=None):
    rclpy.init(args=args)
    node = Nav2GoalBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
