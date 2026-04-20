#!/usr/bin/env python3
"""
Evaluation Runner Node.

Intended purpose:
- Automate the testing process for the OmniVLA model.
- Publish text prompts to trigger the inference node.
- Track robot odometry and collisions during the run.
- Use SuccessChecker and ResultLogger to evaluate performance.
"""

import math
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String

# Import the custom evaluation modules
from omnivla_eval.collision_monitor import CollisionMonitor
from omnivla_eval.metrics import MetricsCalculator
from omnivla_eval.success_checker import SuccessChecker
from omnivla_eval.result_logger import ResultLogger


class EvalRunnerNode(Node):
    def __init__(self):
        super().__init__('eval_runner_node')

        # -- Parameters --
        self.declare_parameter('timeout_sec', 60.0)
        self.declare_parameter('success_threshold_m', 0.5)
        self.declare_parameter('output_dir', 'eval_results')
        
        self.timeout_sec = self.get_parameter('timeout_sec').value
        threshold = self.get_parameter('success_threshold_m').value
        out_dir = self.get_parameter('output_dir').value

        # -- Initialize evaluation tools --
        self.collision_monitor = CollisionMonitor(self)
        self.metrics = MetricsCalculator()
        self.checker = SuccessChecker(threshold=threshold)
        self.logger = ResultLogger(output_dir=out_dir)

        # -- State variables --
        self.current_x = 0.0
        self.current_y = 0.0
        self.is_evaluating = False

        # -- Subscribers & Publishers --
        self.create_subscription(Odometry, '/bcr_bot/odom', self._odom_cb, 10)
        self.prompt_pub = self.create_publisher(String, '/omnivla/prompt', 10)

        self.get_logger().info('EvalRunnerNode is ready.')

    def _odom_cb(self, msg: Odometry):
        """Continuously update the robot's current coordinates and track path if evaluating."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        if self.is_evaluating:
            self.metrics.update_path(self.current_x, self.current_y)

    def run_evaluation_suite(self, test_cases):
        """
        Executes a suite of test cases.
        Expected format for test_cases:
        [
            {"prompt": "go to the left desk", "goal_id": "desk_left", "goal_x": 2.35, "goal_y": -1.10},
            ...
        ]
        """
        self.get_logger().info(f'Starting evaluation suite with {len(test_cases)} tests.')

        for i, task in enumerate(test_cases, 1):
            self.get_logger().info(f'--- Test {i}/{len(test_cases)}: "{task["prompt"]}" ---')
            
            # Optional: Add code here to reset the robot to a starting pose using Nav2
            time.sleep(2.0)  # Wait for the robot to stabilize
            
            start_x, start_y = self.current_x, self.current_y
            goal_x, goal_y = task['goal_x'], task['goal_y']
            start_to_goal_dist = math.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)

            # 1. Start all loggers
            self.metrics.start()
            self.collision_monitor.collision_count = 0
            self.collision_monitor.is_active = True
            self.is_evaluating = True

            # 2. Send the prompt to the model (Triggers inference -> Nav2 goal)
            msg = String()
            msg.data = task['prompt']
            self.prompt_pub.publish(msg)

            # 3. Wait for the test to complete (up to timeout)
            start_time = time.time()
            self.get_logger().info(f'Waiting for navigation (Timeout: {self.timeout_sec}s)...')
            
            while time.time() - start_time < self.timeout_sec:
                rclpy.spin_once(self, timeout_sec=0.1)
                
                # Simple early stopping check: if within success range
                is_success, _ = self.checker.check(self.current_x, self.current_y, goal_x, goal_y)
                if is_success:
                    time.sleep(1.0)  # Ensure it has completely stopped
                    break

            # 4. Stop recording and finalize episode
            self.is_evaluating = False
            self.collision_monitor.is_active = False
            self.metrics.stop()

            is_success, dist_error = self.checker.check(self.current_x, self.current_y, goal_x, goal_y)
            
            record = self.logger.log_episode(
                ep_idx=i,
                prompt=task['prompt'],
                target_id=task['goal_id'],
                is_success=is_success,
                dist_error=dist_error,
                time_s=self.metrics.get_time(),
                path_len=self.metrics.get_length(),
                collisions=self.collision_monitor.collision_count,
                start_to_goal_dist=start_to_goal_dist
            )
            
            outcome_str = "✅ Success" if is_success else "❌ Failure"
            self.get_logger().info(f'Result: {outcome_str} (Error: {dist_error:.2f}m)')

        # 5. Output final report
        report_file, summary = self.logger.save_report()
        self.get_logger().info('================================')
        self.get_logger().info('🎉 Evaluation complete! Report generated.')
        self.get_logger().info(f'Overall Success Rate (SR): {summary["success_rate"]*100:.1f}%')
        self.get_logger().info(f'Path Efficiency (SPL): {summary["average_spl"]:.3f}')
        self.get_logger().info(f'Total Collisions: {summary["total_collisions"]}')
        self.get_logger().info(f'Report saved to: {report_file}')
        self.get_logger().info('================================')


def main(args=None):
    rclpy.init(args=args)
    node = EvalRunnerNode()
    
    # Placeholder for test cases (this can later be loaded from a JSON/YAML file)
    dummy_test_cases = [
        {"prompt": "go to the left desk", "goal_id": "desk_left", "goal_x": 2.35, "goal_y": -1.10},
        {"prompt": "navigate to the right desk", "goal_id": "desk_right", "goal_x": 2.80, "goal_y": 1.15}
    ]
    
    try:
        node.run_evaluation_suite(dummy_test_cases)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
