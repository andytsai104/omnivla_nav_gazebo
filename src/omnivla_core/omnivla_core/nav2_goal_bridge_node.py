"""Nav2 goal bridge node.

Intended purpose:
- receive semantic goal ID or predicted pose from the model
- convert that output into a Nav2-compatible goal
- send the goal to Nav2 for execution
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from .goal_library import GoalLibrary
from .pose_utils import pose_stamped_from_xyyaw


class Nav2GoalBridgeNode(Node):
    def __init__(self):
        super().__init__("nav2_goal_bridge_node")

        self.declare_parameter("input_mode", "goal_id")
        self.declare_parameter("input_goal_id_topic", "/omnivla/inferred_goal_id")
        self.declare_parameter("input_goal_pose_topic", "/omnivla/inferred_goal_pose")
        self.declare_parameter("nav2_goal_topic", "/goal_pose")
        self.declare_parameter("goal_frame_id", "map")
        self.declare_parameter("lookup_from_goal_library", True)
        self.declare_parameter("goal_library_path", "config/goal_library.yaml")
        self.declare_parameter("publish_debug_goal", True)

        self.input_mode = self.get_parameter("input_mode").value
        self.goal_frame_id = self.get_parameter("goal_frame_id").value
        self.lookup_from_goal_library = self.get_parameter("lookup_from_goal_library").value

        input_goal_id_topic = self.get_parameter("input_goal_id_topic").value
        nav2_goal_topic = self.get_parameter("nav2_goal_topic").value
        goal_library_path = self.get_parameter("goal_library_path").value

        self.goal_library = GoalLibrary(goal_library_path)
        self.goal_pub = self.create_publisher(PoseStamped, nav2_goal_topic, 10)
        self._last_goal_id = None

        if self.input_mode == "goal_id":
            self.create_subscription(String, input_goal_id_topic, self._goal_id_cb, 10)
        else:
            self.get_logger().warn("Only input_mode='goal_id' is implemented in this test version.")

        self.get_logger().info("Nav2 goal bridge node started.")

    def _goal_id_cb(self, msg: String) -> None:
        goal_id = msg.data.strip()
        if not goal_id:
            return

        if goal_id == self._last_goal_id:
            return

        if not self.goal_library.has_goal(goal_id):
            self.get_logger().error(f"Unknown goal_id '{goal_id}'")
            return

        goal = self.goal_library.get_goal(goal_id)

        # prefer runtime-configured frame for Nav2
        frame_id = self.goal_frame_id or goal.frame_id

        goal_msg = pose_stamped_from_xyyaw(
            x=goal.pose.x,
            y=goal.pose.y,
            yaw=goal.pose.yaw,
            frame_id=frame_id,
            stamp=self.get_clock().now().to_msg(),
        )
        self.goal_pub.publish(goal_msg)
        self._last_goal_id = goal_id

        self.get_logger().info(
            f"Published Nav2 goal for goal_id='{goal_id}' "
            f"at (x={goal.pose.x:.3f}, y={goal.pose.y:.3f}, yaw={goal.pose.yaw:.3f}) "
            f"in frame '{frame_id}'"
        )


def main(args=None):
    rclpy.init(args=args)
    node = Nav2GoalBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
