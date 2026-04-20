#!/usr/bin/env python3
"""
episode_manager_node.py
-----------------------
Manages the data collection episode lifecycle:

  1. Loads goal_library.yaml
  2. Samples a random goal (random_goal)
  3. Samples a random start pose (random_start), ensuring distance to goal >= min_start_goal_distance
  4. If reset_between_episodes: Uses Nav2 to move the robot to the start pose
  5. Publishes /omnivla/prompt, /omnivla/goal_id, /goal_pose
  6. Calls /omnivla_data/start_logging
  7. Uses Nav2 to navigate the robot to the goal
  8. Waits: Nav2 success -> outcome="success"
            OR episode_length_sec timeout -> outcome="timeout"
  9. Publishes outcome, calls /omnivla_data/stop_logging
  10. Repeats until max_episodes is reached or a stop command is received

Control services:
  /omnivla_data/start_collection  (std_srvs/Trigger) -> Starts collection
  /omnivla_data/stop_collection   (std_srvs/Trigger) -> Stops collection
"""

import json
import math
import random
import threading
import time
from pathlib import Path

import yaml
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from omnivla_data.prompt_sampler import PromptSampler
from omnivla_data.goal_sampler import GoalSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def yaw_to_quaternion(yaw: float) -> dict:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return {"x": 0.0, "y": 0.0, "z": sy, "w": cy}


def make_pose_stamped(x: float, y: float, yaw: float,
                      frame_id: str = 'map') -> PoseStamped:
    q = yaw_to_quaternion(yaw)
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = 0.0
    pose.pose.orientation.x = q['x']
    pose.pose.orientation.y = q['y']
    pose.pose.orientation.z = q['z']
    pose.pose.orientation.w = q['w']
    return pose


def load_goal_library(path: str) -> list:
    """
    Loads goal_library.yaml, returns only goals with enabled=True.
    If the enabled field is missing, defaults to True.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Cannot find goal_library.yaml: {path}')

    with open(p, 'r') as f:
        data = yaml.safe_load(f)

    goals = data.get('goals', [])
    if not goals:
        raise ValueError(f'goal_library.yaml contains no goals: {path}')

    enabled = [g for g in goals if g.get('enabled', True)]
    if not enabled:
        raise ValueError('goal_library.yaml contains no enabled=true goals')

    return enabled


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class EpisodeManagerNode(Node):

    def __init__(self):
        super().__init__('episode_manager_node')

        # ── Parameters (fully aligned with collection.yaml) ────────────────────────────
        self.declare_parameter('goal_library_path',        'config/goal_library.yaml')
        self.declare_parameter('prompt_template_path',     'config/prompt_templates.yaml')
        self.declare_parameter('prompt_mode',              'aliases')
        self.declare_parameter('random_start',             True)
        self.declare_parameter('random_goal',              True)
        self.declare_parameter('reset_between_episodes',   True)
        self.declare_parameter('episode_length_sec',       45.0)
        self.declare_parameter('max_episodes',             50)
        self.declare_parameter('min_start_goal_distance',  1.5)
        self.declare_parameter('spawn_frame_id',           'map')
        self.declare_parameter('publish_prompt_topic',     '/omnivla/prompt')
        self.declare_parameter('publish_goal_id_topic',    '/omnivla/goal_id')
        self.declare_parameter('publish_goal_pose_topic',  '/goal_pose')
        self.declare_parameter('reset_timeout_sec', 60.0)

        self.goal_library_path       = self.get_parameter('goal_library_path').value
        self.prompt_template_path    = self.get_parameter('prompt_template_path').value
        self.prompt_mode             = self.get_parameter('prompt_mode').value
        self.random_start            = self.get_parameter('random_start').value
        self.random_goal             = self.get_parameter('random_goal').value
        self.reset_between_episodes  = self.get_parameter('reset_between_episodes').value
        self.episode_length_sec      = self.get_parameter('episode_length_sec').value
        self.max_episodes            = self.get_parameter('max_episodes').value
        self.min_dist                = self.get_parameter('min_start_goal_distance').value
        self.spawn_frame_id          = self.get_parameter('spawn_frame_id').value
        self.prompt_topic            = self.get_parameter('publish_prompt_topic').value
        self.goal_id_topic           = self.get_parameter('publish_goal_id_topic').value
        self.goal_pose_topic         = self.get_parameter('publish_goal_pose_topic').value
        self.reset_timeout_sec = self.get_parameter('reset_timeout_sec').value

        # ── Goal library ────────────────────────────────────────────────────
        self.goals = load_goal_library(self.goal_library_path)
        self.get_logger().info(
            f'Loaded {len(self.goals)} available goals from: {self.goal_library_path}'
        )

        self.prompt_sampler = PromptSampler(
            mode=self.prompt_mode, 
            template_yaml_path=self.prompt_template_path
        )

        self.goal_sampler = GoalSampler(
            goals=self.goals, 
            min_dist=self.min_dist
        )
        
        # ── State ────────────────────────────────────────────────────────────
        self._running          = False
        self._stop_requested   = False
        self._collection_lock  = threading.Lock()
        self._collection_thread = None

        # ── Nav2 action client ───────────────────────────────────────────────
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ── Service clients (calls data_logger_node) ──────────────────────────
        self._start_log_cli = self.create_client(Trigger, '/omnivla_data/start_logging')
        self._stop_log_cli  = self.create_client(Trigger, '/omnivla_data/stop_logging')

        # ── Publishers ───────────────────────────────────────────────────────
        self._prompt_pub    = self.create_publisher(String,       self.prompt_topic,   10)
        self._goal_id_pub   = self.create_publisher(String,       self.goal_id_topic,  10)
        self._goal_pose_pub = self.create_publisher(PoseStamped,  self.goal_pose_topic, 10)
        self._outcome_pub   = self.create_publisher(String, '/omnivla_data/episode_outcome', 10)
        self._episode_pub   = self.create_publisher(String, '/omnivla/current_episode', 10)

        # ── Control services ─────────────────────────────────────────────────
        self.create_service(
            Trigger,
            '/omnivla_data/start_collection',
            self._handle_start_collection)

        self.create_service(
            Trigger,
            '/omnivla_data/stop_collection',
            self._handle_stop_collection)

        self.get_logger().info(
            'EpisodeManagerNode initialized successfully.\n'
            'Run the following command to start data collection:\n'
            '  ros2 service call /omnivla_data/start_collection std_srvs/srv/Trigger'
        )

    # ── Control service handlers ──────────────────────────────────────────────

    def _handle_start_collection(self, request, response):
        with self._collection_lock:
            if self._running:
                response.success = False
                response.message = 'Collection is already running'
                return response
            self._stop_requested    = False
            self._running           = True
            self._collection_thread = threading.Thread(
                target=self._collection_loop, daemon=True)
            self._collection_thread.start()

        response.success = True
        response.message = f'Started collection, up to {self.max_episodes} episodes'
        self.get_logger().info(response.message)
        return response

    def _handle_stop_collection(self, request, response):
        with self._collection_lock:
            if not self._running:
                response.success = False
                response.message = 'Collection is not currently running'
                return response
            self._stop_requested = True

        response.success = True
        response.message = 'Stop request recorded, will stop after the current episode finishes'
        self.get_logger().info(response.message)
        return response

    # ── Collection loop ───────────────────────────────────────────────────────

    def _collection_loop(self):
        """
        Main loop: completes a full episode each iteration.
        Runs in a background thread.
        """
        for ep_num in range(1, self.max_episodes + 1):

            with self._collection_lock:
                if self._stop_requested:
                    self.get_logger().info('Received stop request, ending collection loop')
                    self._running = False
                    return

            self.get_logger().info(f'══ Episode {ep_num}/{self.max_episodes} Started ══')

            # 1. Sample goal (Target)
            if self.random_goal:
                goal = self.goal_sampler.sample_target()
            else:
                goal = self.goals[0]
                
            prompt = self._sample_prompt(goal)
            self.get_logger().info(
                f'Goal: id="{goal["id"]}"  prompt="{prompt}"'
            )

            # 2. Sample start pose (Initial State), ensuring minimum distance to goal
            start_goal = None
            if self.random_start:
                start_goal = self.goal_sampler.sample_start(target_goal=goal)
                if not start_goal:
                    self.get_logger().warn(
                        f'Could not find a start pose with distance >= {self.min_dist}m, using map origin.'
                    )

            # 3. Reset robot to start pose
            if self.reset_between_episodes:
                if start_goal:
                    sx = start_goal['pose']['x']
                    sy = start_goal['pose']['y']
                    syaw = start_goal['pose']['yaw']
                    fid  = start_goal.get('frame_id', self.spawn_frame_id)
                    self.get_logger().info(
                        f'Resetting to start pose: "{start_goal["id"]}" ({sx}, {sy})'
                    )
                else:
                    # When no candidate start pose, reset to map origin
                    sx, sy, syaw = 0.0, 0.0, 0.0
                    fid = self.spawn_frame_id

                reset_ok = self._navigate_to(
                    sx, sy, syaw, fid,
                    timeout_sec=self.reset_timeout_sec,
                    label='Reset Navigation'
                )
                if not reset_ok:
                    self.get_logger().warn('Reset navigation failed, skipping this episode')
                    continue

            # 4. Publish goal info (received by both data_logger and omnivla_core)
            goal_pose_msg = make_pose_stamped(
                goal['pose']['x'],
                goal['pose']['y'],
                goal['pose']['yaw'],
                goal.get('frame_id', self.spawn_frame_id),
            )
            goal_pose_msg.header.stamp = self.get_clock().now().to_msg()

            self._publish_str(self._prompt_pub,  prompt)
            self._publish_str(self._goal_id_pub, goal['id'])
            self._goal_pose_pub.publish(goal_pose_msg)

            # Publish episode summary (for monitoring by other nodes)
            summary = {
                "episode_num": ep_num,
                "goal_id":     goal['id'],
                "label":       goal['label'],
                "prompt":      prompt,
                "goal_pose":   goal['pose'],
            }
            self._publish_str(self._episode_pub, json.dumps(summary))

            # Brief wait to allow data_logger to receive topics before recording starts
            time.sleep(0.2)

            # 5. Start logging
            start_resp = self._call_trigger(self._start_log_cli, 'start_logging')
            if not start_resp or not start_resp.success:
                self.get_logger().error('start_logging failed, skipping this episode')
                continue

            # 6. Navigate to goal (with episode_length_sec timeout)
            self.get_logger().info(
                f'Starting navigation to goal, timeout={self.episode_length_sec}s'
            )
            nav_ok  = self._navigate_to(
                goal['pose']['x'],
                goal['pose']['y'],
                goal['pose']['yaw'],
                goal.get('frame_id', self.spawn_frame_id),
                timeout_sec=self.episode_length_sec,
                label='Goal Navigation'
            )
            outcome = 'success' if nav_ok else 'timeout'
            self.get_logger().info(f'Episode {ep_num} result: {outcome}')

            # 7. Notify data_logger of the outcome, then stop logging
            self._publish_outcome(outcome)
            time.sleep(0.1)   # Let topic be delivered
            stop_resp = self._call_trigger(self._stop_log_cli, 'stop_logging')
            if not stop_resp or not stop_resp.success:
                self.get_logger().warn('stop_logging call failed')
            
            self.goal_sampler.record_visit(goal['id'])

        self.get_logger().info(
            f'Completed {self.max_episodes} episodes, collection finished'
        )
        with self._collection_lock:
            self._running = False


    # ── Nav2 helper ───────────────────────────────────────────────────────────

    def _navigate_to(self, x: float, y: float, yaw: float,
                     frame_id: str = 'map',
                     timeout_sec: float = 45.0,
                     label: str = 'Navigation') -> bool:
        """
        Sends NavigateToPose goal to Nav2.
        Returns True on success, False on timeout or failure.
        Blocks in the calling thread.
        """
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'[{label}] Nav2 action server unreachable')
            return False

        pose = make_pose_stamped(x, y, yaw, frame_id)
        pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        result_event  = threading.Event()
        result_holder = {"status": None}

        def _response_cb(future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn(f'[{label}] Nav2 goal rejected')
                result_holder["status"] = GoalStatus.STATUS_ABORTED
                result_event.set()
                return

            def _result_cb(res_future):
                result_holder["status"] = res_future.result().status
                result_event.set()

            goal_handle.get_result_async().add_done_callback(_result_cb)

        self._nav_client.send_goal_async(goal_msg).add_done_callback(_response_cb)

        finished = result_event.wait(timeout=timeout_sec)
        if not finished:
            self.get_logger().warn(f'[{label}] Timeout ({timeout_sec}s)')
            return False

        success = result_holder["status"] == GoalStatus.STATUS_SUCCEEDED
        self.get_logger().info(
            f'[{label}] Finished, status={result_holder["status"]}, '
            f'success={success}'
        )
        return success

    # ── Publish helpers ───────────────────────────────────────────────────────

    def _publish_str(self, publisher, text: str):
        msg = String()
        msg.data = text
        publisher.publish(msg)

    def _publish_outcome(self, outcome: str):
        for _ in range(3):   # Publish multiple times to ensure data_logger receives it
            self._publish_str(self._outcome_pub, outcome)
            time.sleep(0.05)

    def _sample_prompt(self, goal: dict) -> str:
        """
        Selects language prompt based on prompt_mode using PromptSampler.
        """

        return self.prompt_sampler.sample(goal)

    # ── Service call helper ───────────────────────────────────────────────────

    def _call_trigger(self, client, name: str):
        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(f'Service "{name}" unavailable')
            return None

        future = client.call_async(Trigger.Request())

        done_event = threading.Event()
        result_holder = {"result": None, "error": None}

        def _done_cb(fut):
            try:
                result_holder["result"] = fut.result()
            except Exception as e:
                result_holder["error"] = e
            finally:
                done_event.set()

        future.add_done_callback(_done_cb)

        finished = done_event.wait(timeout=5.0)
        if not finished:
            self.get_logger().error(f'Service call "{name}" timeout')
            return None

        if result_holder["error"] is not None:
            self.get_logger().error(
                f'Service call "{name}" failed: {result_holder["error"]}'
            )
            return None

        return result_holder["result"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = EpisodeManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()