#!/usr/bin/env python3
"""
data_logger_node.py
-------------------
Subscribes to BCR Bot sensor topics and saves data to disk during an episode.

Controlled by episode_manager_node via two services:
  /omnivla_data/start_logging  (std_srvs/Trigger)  -> Starts a new episode
  /omnivla_data/stop_logging   (std_srvs/Trigger)  -> Ends and saves the episode

Subscribed topics (aligned with collection.yaml):
  - /bcr_bot/stereo_camera/left/image_raw   (sensor_msgs/Image)
  - /bcr_bot/stereo_camera/left/camera_info (sensor_msgs/CameraInfo)
  - /bcr_bot/odom                           (nav_msgs/Odometry)
  - /omnivla/prompt                         (std_msgs/String)
  - /omnivla/goal_id                        (std_msgs/String)
  - /goal_pose                              (geometry_msgs/PoseStamped)

Save format:
  <output_dir>/
    episode_<NNNN>/
      metadata.json
      frames.csv          (if save_metadata_csv: true)
      frames/
        0000.png
        0000.json
        ...
"""

import os
import csv
import json
import math
import time
import threading
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

try:
    from cv_bridge import CvBridge
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def odom_to_dict(msg: Odometry) -> dict:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    v = msg.twist.twist.linear
    w = msg.twist.twist.angular
    return {
        "position":         {"x": p.x, "y": p.y, "z": p.z},
        "orientation_quat": {"x": q.x, "y": q.y, "z": q.z, "w": q.w},
        "yaw_rad":          yaw_from_quaternion(q),
        "linear_vel":       {"x": v.x, "y": v.y},
        "angular_vel_z":    w.z,
    }


def pose_stamped_to_dict(msg: PoseStamped) -> dict:
    p = msg.pose.position
    q = msg.pose.orientation
    return {
        "frame_id": msg.header.frame_id,
        "x":        p.x,
        "y":        p.y,
        "yaw_rad":  yaw_from_quaternion(q),
    }


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


# ---------------------------------------------------------------------------
# Simple time synchronization buffer
# ---------------------------------------------------------------------------

class SyncBuffer:
    """
    Stores the latest message for each key.
    get_synced() returns a dict of messages when the timestamp difference of all keys is <= slop_sec.
    """

    def __init__(self, keys: list, slop_sec: float):
        self._keys   = keys
        self._slop   = slop_sec
        self._msgs   = {k: None for k in keys}
        self._stamps = {k: 0.0  for k in keys}
        self._lock   = threading.Lock()

    def update(self, key: str, msg, stamp_sec: float):
        with self._lock:
            self._msgs[key]   = msg
            self._stamps[key] = stamp_sec

    def get_synced(self):
        """
        If the timestamps of all messages are within the slop range, returns a {key: msg} dict;
        Otherwise, returns None.
        """
        with self._lock:
            if any(m is None for m in self._msgs.values()):
                return None
            stamps = list(self._stamps.values())
            if max(stamps) - min(stamps) > self._slop:
                return None
            return dict(self._msgs)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class DataLoggerNode(Node):

    def __init__(self):
        super().__init__('data_logger_node')

        # ── Parameters (fully aligned with collection.yaml) ────────────────────────────
        self.declare_parameter('output_dir',          'datasets/run_001')
        self.declare_parameter('save_images',         True)
        self.declare_parameter('save_metadata_json',  True)
        self.declare_parameter('save_metadata_csv',   False)
        self.declare_parameter('image_topic',         '/bcr_bot/stereo_camera/left/image_raw')
        self.declare_parameter('camera_info_topic',   '/bcr_bot/stereo_camera/left/camera_info')
        self.declare_parameter('odom_topic',          '/bcr_bot/odom')
        self.declare_parameter('prompt_topic',        '/omnivla/prompt')
        self.declare_parameter('goal_id_topic',       '/omnivla/goal_id')
        self.declare_parameter('goal_pose_topic',     '/goal_pose')
        self.declare_parameter('sample_rate_hz',      2.0)
        self.declare_parameter('sync_slop_sec',       0.20)

        self.output_dir        = self.get_parameter('output_dir').value
        self.save_images       = self.get_parameter('save_images').value
        self.save_json         = self.get_parameter('save_metadata_json').value
        self.save_csv          = self.get_parameter('save_metadata_csv').value
        self.image_topic       = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.odom_topic        = self.get_parameter('odom_topic').value
        self.prompt_topic      = self.get_parameter('prompt_topic').value
        self.goal_id_topic     = self.get_parameter('goal_id_topic').value
        self.goal_pose_topic   = self.get_parameter('goal_pose_topic').value
        self.sample_rate_hz    = self.get_parameter('sample_rate_hz').value
        self.sync_slop_sec     = self.get_parameter('sync_slop_sec').value

        os.makedirs(self.output_dir, exist_ok=True)

        # ── cv_bridge ────────────────────────────────────────────────────────
        self._bridge = CvBridge() if CV_AVAILABLE else None
        if not CV_AVAILABLE:
            self.get_logger().warn(
                'cv_bridge/OpenCV is unavailable — images will be saved in .bin format'
            )

        # ── State ────────────────────────────────────────────────────────────
        self._lock           = threading.Lock()
        self._logging        = False
        self._episode_dir    = None
        self._frames_dir     = None
        self._frame_idx      = 0
        self._episode_meta   = {}
        self._csv_writer     = None
        self._csv_file       = None
        self._episode_count  = self._count_existing_episodes()

        # Asynchronous state (prompt / goal_id / goal_pose do not need to be synchronized with image)
        self._latest_prompt    = ''
        self._latest_goal_id   = ''
        self._latest_goal_pose = None   # PoseStamped

        # ── Sync buffer: image + odom ─────────────────────────────────────────
        self._sync = SyncBuffer(
            keys=['image', 'odom'],
            slop_sec=self.sync_slop_sec,
        )

        # ── QoS ─────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(
            Image, self.image_topic,
            self._image_cb, sensor_qos)

        self.create_subscription(
            CameraInfo, self.camera_info_topic,
            self._camera_info_cb, sensor_qos)

        self.create_subscription(
            Odometry, self.odom_topic,
            self._odom_cb, sensor_qos)

        self.create_subscription(
            String, self.prompt_topic,
            self._prompt_cb, 10)

        self.create_subscription(
            String, self.goal_id_topic,
            self._goal_id_cb, 10)

        self.create_subscription(
            PoseStamped, self.goal_pose_topic,
            self._goal_pose_cb, 10)

        # Result sent by episode_manager before ending
        self.create_subscription(
            String, '/omnivla_data/episode_outcome',
            self._outcome_cb, 10)

        # ── Services ─────────────────────────────────────────────────────────
        self.create_service(
            Trigger,
            '/omnivla_data/start_logging',
            self._handle_start)

        self.create_service(
            Trigger,
            '/omnivla_data/stop_logging',
            self._handle_stop)

        # ── Logging timer ─────────────────────────────────────────────────────
        self.create_timer(1.0 / self.sample_rate_hz, self._log_tick)

        self.get_logger().info(
            f'DataLoggerNode initialized successfully\n'
            f'  output_dir   : {self.output_dir}\n'
            f'  sample_rate  : {self.sample_rate_hz} Hz\n'
            f'  sync_slop    : {self.sync_slop_sec} s\n'
            f'  image_topic  : {self.image_topic}\n'
            f'  save_csv     : {self.save_csv}'
        )

    # ── Sensor callbacks ─────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        t = stamp_to_sec(msg.header.stamp)
        self._sync.update('image', msg, t)

    def _camera_info_cb(self, msg: CameraInfo):
        # Reserved for future use (e.g., intrinsics for 3D reconstruction)
        pass

    def _odom_cb(self, msg: Odometry):
        t = stamp_to_sec(msg.header.stamp)
        self._sync.update('odom', msg, t)

    def _prompt_cb(self, msg: String):
        with self._lock:
            self._latest_prompt = msg.data

    def _goal_id_cb(self, msg: String):
        with self._lock:
            self._latest_goal_id = msg.data

    def _goal_pose_cb(self, msg: PoseStamped):
        with self._lock:
            self._latest_goal_pose = msg

    def _outcome_cb(self, msg: String):
        """Receives the outcome from episode_manager: success / timeout / aborted"""
        with self._lock:
            if self._episode_meta:
                self._episode_meta['outcome'] = msg.data

    # ── Service handlers ─────────────────────────────────────────────────────

    def _handle_start(self, request, response):
        with self._lock:
            if self._logging:
                response.success = False
                response.message = 'Already logging, please call stop_logging first'
                return response

            self._episode_count += 1
            ep_name           = f'episode_{self._episode_count:04d}'
            self._episode_dir = os.path.join(self.output_dir, ep_name)
            self._frames_dir  = os.path.join(self._episode_dir, 'frames')
            os.makedirs(self._frames_dir, exist_ok=True)

            self._frame_idx    = 0
            self._episode_meta = {
                "episode_id":  ep_name,
                "start_time":  datetime.utcnow().isoformat() + 'Z',
                "end_time":    None,
                "outcome":     None,
                "goal_id":     self._latest_goal_id,
                "prompt":      self._latest_prompt,
                "goal_pose":   (pose_stamped_to_dict(self._latest_goal_pose)
                                if self._latest_goal_pose else None),
                "frame_count": 0,
            }

            # Open CSV (if enabled)
            if self.save_csv:
                csv_path = os.path.join(self._episode_dir, 'frames.csv')
                self._csv_file   = open(csv_path, 'w', newline='')
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow([
                    'frame_idx', 'ros_stamp_sec',
                    'pos_x', 'pos_y', 'yaw_rad',
                    'linear_vx', 'angular_vz',
                    'prompt', 'goal_id',
                ])

            self._logging = True

        response.success = True
        response.message = f'Started logging -> {self._episode_dir}'
        self.get_logger().info(response.message)
        return response

    def _handle_stop(self, request, response):
        with self._lock:
            if not self._logging:
                response.success = False
                response.message = 'Currently not logging'
                return response
            self._logging = False
            self._finalise_episode()

        response.success = True
        response.message = f'Stopped logging -> {self._episode_dir}'
        self.get_logger().info(response.message)
        return response

    # ── Logging timer ─────────────────────────────────────────────────────────

    def _log_tick(self):
        """Triggered at sample_rate_hz frequency to attempt saving a synchronized frame."""
        with self._lock:
            if not self._logging:
                return

        synced = self._sync.get_synced()
        if synced is None:
            return   # image and odom are not time-aligned yet, skipping this tick

        img_msg  = synced['image']
        odom_msg = synced['odom']

        with self._lock:
            idx       = self._frame_idx
            prompt    = self._latest_prompt
            goal_id   = self._latest_goal_id
            goal_pose = self._latest_goal_pose

        # --- Save Image ---
        if self.save_images:
            img_path = os.path.join(self._frames_dir, f'{idx:04d}.png')
            if CV_AVAILABLE and self._bridge:
                try:
                    cv_img = self._bridge.imgmsg_to_cv2(img_msg, 'bgr8')
                    cv2.imwrite(img_path, cv_img)
                except Exception as e:
                    self.get_logger().warn(f'Failed to save image: {e}')
                    return
            else:
                with open(img_path.replace('.png', '.bin'), 'wb') as f:
                    f.write(bytes(img_msg.data))

        # --- Construct frame data ---
        odom_dict = odom_to_dict(odom_msg)
        ros_t     = stamp_to_sec(img_msg.header.stamp)

        frame_data = {
            "frame_idx":  idx,
            "ros_stamp":  ros_t,
            "wall_time":  time.time(),
            "prompt":     prompt,
            "goal_id":    goal_id,
            "goal_pose":  pose_stamped_to_dict(goal_pose) if goal_pose else None,
            "odom":       odom_dict,
        }

        # --- Save JSON ---
        if self.save_json:
            json_path = os.path.join(self._frames_dir, f'{idx:04d}.json')
            with open(json_path, 'w') as f:
                json.dump(frame_data, f, indent=2)

        # --- Save CSV row ---
        if self.save_csv and self._csv_writer:
            self._csv_writer.writerow([
                idx,
                round(ros_t, 4),
                round(odom_dict['position']['x'], 4),
                round(odom_dict['position']['y'], 4),
                round(odom_dict['yaw_rad'], 4),
                round(odom_dict['linear_vel']['x'], 4),
                round(odom_dict['angular_vel_z'], 4),
                prompt,
                goal_id,
            ])

        with self._lock:
            self._frame_idx += 1

    # ── Finalise ──────────────────────────────────────────────────────────────

    def _finalise_episode(self):
        """Saves metadata.json and closes CSV. Must be called while holding the lock."""
        if self._episode_meta.get('outcome') is None:
            self._episode_meta['outcome'] = 'aborted'

        self._episode_meta['end_time']    = datetime.utcnow().isoformat() + 'Z'
        self._episode_meta['frame_count'] = self._frame_idx

        if self.save_json:
            meta_path = os.path.join(self._episode_dir, 'metadata.json')
            with open(meta_path, 'w') as f:
                json.dump(self._episode_meta, f, indent=2)

        if self._csv_file:
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None

        self.get_logger().info(
            f'[{self._episode_meta["episode_id"]}] Save completed | '
            f'outcome={self._episode_meta["outcome"]} | '
            f'frames={self._frame_idx}'
        )

    # ── Helper ───────────────────────────────────────────────────────────────

    def _count_existing_episodes(self) -> int:
        if not os.path.isdir(self.output_dir):
            return 0
        return len([
            d for d in os.listdir(self.output_dir)
            if d.startswith('episode_')
        ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = DataLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()