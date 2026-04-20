"""Evaluation metrics.

Intended purpose:
- compute success rate
- compute navigation time
- compute collision count / collision rate
- support final report plots
"""

import time
import math

class MetricsCalculator:
    def __init__(self):
        self.start_t = 0.0
        self.end_t = 0.0
        self.path = []

    def start(self):
        self.start_t = time.time()
        self.path = []

    def stop(self):
        self.end_t = time.time()

    def update_path(self, x, y):
        if not self.path or math.sqrt((x-self.path[-1][0])**2 + (y-self.path[-1][1])**2) > 0.1:
            self.path.append((x, y))

    def get_time(self):
        return self.end_t - self.start_t

    def get_length(self):
        l = 0.0
        for i in range(1, len(self.path)):
            l += math.sqrt((self.path[i][0]-self.path[i-1][0])**2 + (self.path[i][1]-self.path[i-1][1])**2)
        return l
