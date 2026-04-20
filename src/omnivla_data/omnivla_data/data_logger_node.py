"""Data logger node.

Intended purpose:
- subscribe to image / pose / prompt / goal topics
- save synchronized samples for training
- write metadata to disk in a format that is easy to export later

example format for goal_library.yaml:
goals:
  - id: internel key for nav2, omnivla model, evaluation node...
    label: display-friendly words
    category: optional, but 
    frame_id: "map" (ros2 coordinate frame, Nav2 default use "map")
    pose: robot's pose get from ros2 topic (should be align with the label and id)
      x: X.X
      y: X.X
      yaw: X.X
    goal_image: the path for sampled goal image
    aliases: guidance for lanuage-guided navigation (natural words for this pose)
      - "left desk"
      - "desk on the left"
      - "go to the left desk"
    room: optional since we only have one scene in warehouse
    enabled: boolean for whther to enable this label (optional but keep it for larger dataset)

  - id: desk_left
    label: "left desk"
    category: "workspace"
    frame_id: "map"
    pose:
      x: 2.35
      y: -1.10
      yaw: 1.57
    goal_image: "assets/goal_images/desk_left.png"
    aliases:
      - "left desk"
      - "desk on the left"
      - "go to the left desk"
    room: "warehouse_room_a"
    enabled: true

  - id: desk_right
    label: "right desk"
    category: "workspace"
    frame_id: "map"
    pose:
      x: 2.80
      y: 1.15
      yaw: -1.57
    goal_image: "assets/goal_images/desk_right.png"
    aliases:
      - "right desk"
      - "desk on the right"
      - "go to the right desk"
    room: "warehouse_room_a"
    enabled: true
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
