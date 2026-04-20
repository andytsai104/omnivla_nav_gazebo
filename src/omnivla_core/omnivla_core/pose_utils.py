"""Pose utilities.

Intended purpose:
- convert between odom / map / robot frames as needed
- normalize yaw
- help compute relative or target poses
"""

#!/usr/bin/env python3
from __future__ import annotations

import math

from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from nav_msgs.msg import Odometry


def normalize_yaw(yaw: float) -> float:
    return math.atan2(math.sin(yaw), math.cos(yaw))


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def quaternion_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def odom_to_xyyaw(msg: Odometry) -> tuple[float, float, float]:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return float(p.x), float(p.y), float(quaternion_to_yaw(q))


def pose_stamped_from_xyyaw(
    x: float,
    y: float,
    yaw: float,
    frame_id: str,
    stamp,
) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.position.z = 0.0
    msg.pose.orientation = yaw_to_quaternion(yaw)
    return msg


def distance_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)
