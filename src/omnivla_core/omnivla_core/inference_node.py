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

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter("model_type", "omnivla_edge")
        self.declare_parameter("mode", "prediction")   # pipeline_check | prediction
        self.declare_parameter("device", "cuda")

        self.declare_parameter(
            "base_model_checkpoint_path",
            "omnivla-edge/omnivla-edge.pth"
        )
        self.declare_parameter(
            "classifier_checkpoint_path",
            "omnivla_finetune/checkpoints/omnivla_edge_goal_classifier.pt"
        )

        self.declare_parameter("inference_rate_hz", 2.0)
        self.declare_parameter("confidence_threshold", 0.50)

        self.declare_parameter("image_topic", "/bcr_bot/stereo_camera/left/image_raw")
        self.declare_parameter("odom_topic", "/bcr_bot/odom")
        self.declare_parameter("prompt_topic", "/omnivla/prompt")
        self.declare_parameter("output_goal_id_topic", "/omnivla/inferred_goal_id")
        self.declare_parameter("goal_library_path", "./src/omnivla_bringup/config/goal_library.yaml")

        # ------------------------------------------------------------------
        # Read parameters
        # ------------------------------------------------------------------
        self.model_type = self.get_parameter("model_type").value
        self.mode = self.get_parameter("mode").value
        self.device = self.get_parameter("device").value
        self.base_model_checkpoint_path = self.get_parameter("base_model_checkpoint_path").value
        self.classifier_checkpoint_path = self.get_parameter("classifier_checkpoint_path").value
        self.inference_rate_hz = float(self.get_parameter("inference_rate_hz").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        

        # in inference_node.py __init__
        self._active_goal_id = None
        self._active_goal_confidence = 0.0
        self._min_switch_margin = 0.20
        self._blocked_goal_ids = {"home_position"}

        image_topic = self.get_parameter("image_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        prompt_topic = self.get_parameter("prompt_topic").value
        output_goal_id_topic = self.get_parameter("output_goal_id_topic").value
        goal_library_path = self.get_parameter("goal_library_path").value

        # ------------------------------------------------------------------
        # Load goal library + model client
        # ------------------------------------------------------------------
        self.goal_library = GoalLibrary(goal_library_path)

        self.model = ModelClient(
            model_type=self.model_type,
            base_model_checkpoint_path=self.base_model_checkpoint_path,
            classifier_checkpoint_path=self.classifier_checkpoint_path,
            device=self.device,
            goal_library=self.goal_library,
        )

        if self.mode == "prediction":
            if self.model.model_loaded:
                self.get_logger().info("Prediction mode enabled: finetuned classifier loaded.")
            else:
                self.get_logger().warn(
                    "Prediction mode enabled, but model loading failed. "
                    f"Will fallback to rule-based behavior. Reason: {self.model.load_error}"
                )
        elif self.mode == "pipeline_check":
            self.get_logger().info("Pipeline-check mode enabled: using rule-based goal prediction.")
        else:
            self.get_logger().warn(
                f"Unknown mode '{self.mode}'. Supported: pipeline_check, prediction. "
                "Node will still run but inference will reject until mode is corrected."
            )

        # ------------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------------
        self._latest_image = None
        self._latest_pose = None
        self._latest_prompt = None
        self._last_goal_id_sent = None

        # ------------------------------------------------------------------
        # Subscribers / publishers
        # ------------------------------------------------------------------
        self.create_subscription(Image, image_topic, self._image_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.create_subscription(String, prompt_topic, self._prompt_cb, 10)

        self.goal_id_pub = self.create_publisher(String, output_goal_id_topic, 10)

        # ------------------------------------------------------------------
        # Timer
        # ------------------------------------------------------------------
        period = 1.0 / max(0.1, self.inference_rate_hz)
        self.timer = self.create_timer(period, self._run_inference)

        self.get_logger().info(
            f"Inference node started | mode={self.mode} | rate={self.inference_rate_hz:.2f} Hz"
        )

    def _image_cb(self, msg: Image) -> None:
        try:
            self._latest_image = preprocess_image(ros_image_to_bgr(msg))
        except Exception as e:
            self.get_logger().warn(f"Image preprocess failed: {e}")

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_pose = odom_to_xyyaw(msg)

    def _prompt_cb(self, msg: String) -> None:
        self._latest_prompt = msg.data.strip()

    def _run_inference(self) -> None:
        if self.mode not in {"pipeline_check", "prediction"}:
            self.get_logger().warn(
                f"Unsupported mode '{self.mode}'. Supported modes: pipeline_check, prediction"
            )
            return

        if self._latest_image is None or self._latest_pose is None or not self._latest_prompt:
            return

        goal_id, confidence = self.model.predict_goal_id(
            image=self._latest_image,
            pose_xyyaw=self._latest_pose,
            prompt=self._latest_prompt,
            goal_image=None,
            mode=self.mode,
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
        
        # Ignore home_position unless explicitly asked
        if goal_id in self._blocked_goal_ids and "home" not in self._latest_prompt.lower():
            # self.get_logger().warn(f"Ignoring accidental goal_id='{goal_id}'")
            return

        # If we already have a goal, only switch if confidence is clearly higher
        if self._active_goal_id is not None and goal_id != self._active_goal_id:
            if confidence < self._active_goal_confidence + self._min_switch_margin:
                self.get_logger().warn(
                    f"Ignoring goal switch {self._active_goal_id} -> {goal_id}: "
                    f"{confidence:.2f} not higher than {self._active_goal_confidence:.2f} + margin"
                )
                return

        self._active_goal_id = goal_id
        self._active_goal_confidence = confidence

        msg = String()
        msg.data = goal_id
        self.goal_id_pub.publish(msg)
        self._last_goal_id_sent = goal_id

        self.get_logger().info(
            f"[{self.mode}] Published inferred goal_id='{goal_id}' "
            f"with confidence={confidence:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()