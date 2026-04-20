"""Success checker.

Intended purpose:
- decide if a trial reached the correct goal
- check distance / orientation thresholds
"""

import math

class SuccessChecker:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def check(self, curr_x, curr_y, goal_x, goal_y):
        dist = math.sqrt((goal_x - curr_x)**2 + (goal_y - curr_y)**2)
        return dist <= self.threshold, dist
