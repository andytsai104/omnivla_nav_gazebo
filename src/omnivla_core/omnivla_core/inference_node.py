"""Runtime inference node.

Intended purpose:
- subscribe to current RGB image / pose / prompt
- preprocess inputs for OmniVLA / OmniVLA-edge
- run inference
- output either:
  1) semantic goal ID, or
  2) target pose
"""

#!/usr/bin/env python3
from __future__ import annotations

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .goal_library import GoalLibrary
from .image_utils import preprocess_image, ros_image_to_bgr
from .model_client import ModelClient
from .pose_utils import odom_to_xyyaw


class InferenceNode(Node):
    def __init__(self):
        super().__init__("inference_node")

        self.declare_parameter("model_type", "omnivla_edge")
        self.declare_parameter("mode", "goal_id")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("checkpoint_path", "checkpoints/latest.pt")
        self.declare_parameter("inference_rate_hz", 2.0)
        self.declare_parameter("confidence_threshold", 0.50)

        self.declare_parameter("image_topic", "/bcr_bot/stereo_camera/left/image_raw")
        self.declare_parameter("odom_topic", "/bcr_bot/odom")
        self.declare_parameter("prompt_topic", "/omnivla/prompt")
        self.declare_parameter("output_goal_id_topic", "/omnivla/inferred_goal_id")
        self.declare_parameter("goal_library_path", "config/goal_library.yaml")

        self.mode = self.get_parameter("mode").get_parameter_value().string_value
        self.confidence_threshold = (
            self.get_parameter("confidence_threshold").get_parameter_value().double_value
        )

        image_topic = self.get_parameter("image_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        prompt_topic = self.get_parameter("prompt_topic").value
        output_goal_id_topic = self.get_parameter("output_goal_id_topic").value
        goal_library_path = self.get_parameter("goal_library_path").value

        self.goal_library = GoalLibrary(goal_library_path)
        self.model = ModelClient(
            model_type=self.get_parameter("model_type").value,
            checkpoint_path=self.get_parameter("checkpoint_path").value,
            device=self.get_parameter("device").value,
            goal_library=self.goal_library,
        )

        self._latest_image = None
        self._latest_pose = None
        self._latest_prompt = None
        self._last_goal_id_sent = None

        self.create_subscription(Image, image_topic, self._image_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.create_subscription(String, prompt_topic, self._prompt_cb, 10)

        self.goal_id_pub = self.create_publisher(String, output_goal_id_topic, 10)

        period = 1.0 / max(0.1, float(self.get_parameter("inference_rate_hz").value))
        self.timer = self.create_timer(period, self._run_inference)

        self.get_logger().info("Inference node started.")

    def _image_cb(self, msg: Image) -> None:
        try:
            self._latest_image = preprocess_image(ros_image_to_bgr(msg))
        except Exception as e:
            self.get_logger().warn(f"Image preprocess failed: {e}")

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_pose = odom_to_xyyaw(msg)

    def _prompt_cb(self, msg: String) -> None:
        self._latest_prompt = msg.data

    def _run_inference(self) -> None:
        if self.mode != "goal_id":
            self.get_logger().warn("Only mode='goal_id' is implemented in this test version.")
            return

        if self._latest_image is None or self._latest_pose is None or not self._latest_prompt:
            return

        goal_id, confidence = self.model.predict_goal_id(
            image=self._latest_image,
            pose_xyyaw=self._latest_pose,
            prompt=self._latest_prompt,
            goal_image=None,
        )

        if not goal_id:
            self.get_logger().warn("No goal predicted from prompt.")
            return

        if confidence < self.confidence_threshold:
            self.get_logger().warn(
                f"Predicted goal '{goal_id}' below confidence threshold: {confidence:.2f}"
            )
            return

        if goal_id == self._last_goal_id_sent:
            return

        msg = String()
        msg.data = goal_id
        self.goal_id_pub.publish(msg)
        self._last_goal_id_sent = goal_id
        self.get_logger().info(
            f"Published inferred goal_id='{goal_id}' with confidence={confidence:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
