"""Evaluation runner node.

Intended purpose:
- run multiple trials automatically
- trigger episodes / prompts / goals
- collect outcome metrics
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from .success_checker import SuccessChecker
from .metrics import MetricsCalculator
from .collision_monitor import CollisionMonitor
from .result_logger import ResultLogger

class EvalRunnerNode(Node):
    def __init__(self):
        super().__init__('eval_runner_node')
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.checker = SuccessChecker()
        self.metrics = MetricsCalculator()
        self.monitor = CollisionMonitor(self)
        self.logger = ResultLogger()
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.curr_pose = None
        self.goals = [(2.0, 1.5), (-3.0, 4.0), (5.0, -2.0)]
        self.idx = 0
        self.timer = self.create_timer(2.0, self.control_loop)

    def odom_cb(self, msg):
        self.curr_pose = msg.pose.pose.position
        if self.monitor.is_active:
            self.metrics.update_path(self.curr_pose.x, self.curr_pose.y)

    def control_loop(self):
        if self.idx >= len(self.goals) or self.curr_pose is None:
            return
        self.timer.cancel()
        self.run_task()

    def run_task(self):
        gx, gy = self.goals[self.idx]
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = gx
        goal.pose.pose.position.y = gy
        goal.pose.pose.orientation.w = 1.0
        self.monitor.is_active = True
        self.metrics.start()
        if not self.client.wait_for_server(timeout_sec=2.0):
            return
        self.future = self.client.send_goal_async(goal)
        self.future.add_done_callback(self.response_cb)

    def response_cb(self, future):
        handle = future.result()
        if handle.accepted:
            self.res_future = handle.get_result_async()
            self.res_future.add_done_callback(self.done_cb)

    def done_cb(self, future):
        self.metrics.stop()
        self.monitor.is_active = False
        gx, gy = self.goals[self.idx]
        s, d = self.checker.check(self.curr_pose.x, self.curr_pose.y, gx, gy)
        self.logger.log(self.idx, s, self.metrics.get_time(), self.metrics.get_length(), self.monitor.collision_count)
        self.idx += 1
        self.monitor.collision_count = 0
        self.timer = self.create_timer(2.0, self.control_loop)

def main(args=None):
    rclpy.init(args=args)
    node = EvalRunnerNode()
    rclpy.spin(node)
    rclpy.shutdown()
