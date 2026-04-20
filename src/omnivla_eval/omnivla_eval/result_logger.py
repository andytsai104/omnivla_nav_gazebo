"""Result logger.

Intended purpose:
- save per-trial evaluation outputs
- write CSV / JSON summaries for later plotting
"""

import csv
import os

class ResultLogger:
    def __init__(self, path="eval_results.csv"):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Success", "Time", "Length", "Collisions"])

    def log(self, tid, success, t, l, c):
        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([tid, success, round(t, 2), round(l, 2), c])
