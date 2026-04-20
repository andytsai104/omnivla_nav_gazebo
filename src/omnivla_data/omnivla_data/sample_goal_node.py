#!/usr/bin/env python3

import math
from pathlib import Path
from typing import Optional

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class SampleGoalNode(Node):
    def __init__(self):
        super().__init__("sample_goal_node")

        # Goal-specific params: usually overridden from CLI
        self.declare_parameter("goal_id", "desk_left")
        self.declare_parameter("label", "left desk")
        self.declare_parameter("category", "workspace")
        self.declare_parameter("aliases_csv", "")
        self.declare_parameter("image_filename", "")

        # Stable defaults: better loaded from YAML
        self.declare_parameter("image_topic", "/bcr_bot/stereo_camera/left/image_raw")
        self.declare_parameter("odom_topic", "/bcr_bot/odom")
        self.declare_parameter("image_output_dir", "src/omnivla_bringup/assets/goal_images")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("room", "warehouse")
        self.declare_parameter("enabled", True)
        self.declare_parameter("xy_round_digits", 3)
        self.declare_parameter("yaw_round_digits", 3)

        self.goal_id = self.get_parameter("goal_id").value
        self.label = self.get_parameter("label").value
        self.category = self.get_parameter("category").value
        self.aliases_csv = self.get_parameter("aliases_csv").value
        self.image_filename = self.get_parameter("image_filename").value

        self.image_topic = self.get_parameter("image_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.image_output_dir = self.get_parameter("image_output_dir").value
        self.frame_id = self.get_parameter("frame_id").value
        self.room = self.get_parameter("room").value
        self.enabled = self.get_parameter("enabled").value
        self.xy_round_digits = int(self.get_parameter("xy_round_digits").value)
        self.yaw_round_digits = int(self.get_parameter("yaw_round_digits").value)

        self.aliases = [a.strip() for a in self.aliases_csv.split(",") if a.strip()]
        if not self.aliases:
            self.aliases = [self.label, f"go to the {self.label}"]

        if not self.image_filename:
            self.image_filename = f"{self.goal_id}.png"

        self.bridge = CvBridge()
        self.latest_image: Optional[Image] = None
        self.latest_odom: Optional[Odometry] = None
        self.saved_image_relpath = f"assets/goal_images/{self.image_filename}"

        self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.timer = self.create_timer(0.2, self.try_finalize)

        self.get_logger().info(
            f"Waiting for one image from {self.image_topic} and one odom from {self.odom_topic}..."
        )

    def image_callback(self, msg: Image):
        if self.latest_image is None:
            self.latest_image = msg
            self.get_logger().info("Received image.")

    def odom_callback(self, msg: Odometry):
        if self.latest_odom is None:
            self.latest_odom = msg
            self.get_logger().info("Received odom.")

    def try_finalize(self):
        if self.latest_image is None or self.latest_odom is None:
            return

        self.save_image()
        self.print_yaml_entry()
        self.get_logger().info("Done.")
        rclpy.shutdown()

    def save_image(self):
        output_dir = Path(self.image_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / self.image_filename

        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding="bgr8")
        except Exception:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image)

        ok = cv2.imwrite(str(output_path), cv_image)
        if not ok:
            raise RuntimeError(f"Failed to save image to {output_path}")

        self.get_logger().info(f"Saved image to {output_path}")

    def print_yaml_entry(self):
        odom = self.latest_odom
        pos = odom.pose.pose.position
        ori = odom.pose.pose.orientation

        x = round(pos.x, self.xy_round_digits)
        y = round(pos.y, self.xy_round_digits)
        yaw = round(
            yaw_from_quaternion(ori.x, ori.y, ori.z, ori.w),
            self.yaw_round_digits,
        )

        alias_lines = "\n".join([f'      - "{a}"' for a in self.aliases])

        yaml_block = f"""
  - id: {self.goal_id}
    label: "{self.label}"
    category: "{self.category}"
    frame_id: "{self.frame_id}"
    pose:
      x: {x}
      y: {y}
      yaw: {yaw}
    goal_image: "{self.saved_image_relpath}"
    aliases:
{alias_lines}
    room: "{self.room}"
    enabled: {str(self.enabled).lower()}
""".rstrip()

        print("\nCopy this into goal_library.yaml under 'goals:'\n")
        print(yaml_block)
        print("")


def main(args=None):
    rclpy.init(args=args)
    node = SampleGoalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()