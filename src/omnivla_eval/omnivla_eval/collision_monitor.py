"""Collision monitor.

Intended purpose:
- monitor contact or collision-related topics
- log collision events during evaluation
"""

from sensor_msgs.msg import LaserScan

class CollisionMonitor:
    def __init__(self, node):
        self.node = node
        self.collision_count = 0
        self.min_dist = 0.25
        self.is_active = False
        self.sub = self.node.create_subscription(LaserScan, '/bcr_bot/scan', self.callback, 10)

    def callback(self, msg):
        if not self.is_active:
            return
        valid_ranges = [r for r in msg.ranges if r > 0.01 and r < 10.0]
        if valid_ranges and min(valid_ranges) < self.min_dist:
            self.collision_count += 1
