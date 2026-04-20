#!/usr/bin/env python3
"""
Result logger.

Intended purpose:
- aggregate data from metrics, success_checker, and collision_monitor
- calculate SPL (Success weighted by Path Length)
- export a final JSON/CSV report for the entire evaluation run
"""

import json
import os
from datetime import datetime

class ResultLogger:
    def __init__(self, output_dir="eval_results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_episode(self, ep_idx, prompt, target_id, is_success, dist_error, time_s, path_len, collisions, start_to_goal_dist):
        """
        Logs the result of a single test episode and calculates its SPL.
        SPL Formula: S * (l / max(p, l))
        S = 1 (Success) or 0 (Failure), l = shortest straight-line distance, p = actual path length taken
        """
        spl = 0.0
        if is_success:
            # Protection against division by zero
            max_dist = max(start_to_goal_dist, path_len)
            spl = (start_to_goal_dist / max_dist) if max_dist > 0 else 1.0

        record = {
            "episode": ep_idx,
            "prompt": prompt,
            "target_goal_id": target_id,
            "success": bool(is_success),
            "distance_error_m": round(dist_error, 3),
            "time_taken_s": round(time_s, 2),
            "path_length_m": round(path_len, 3),
            "spl": round(spl, 3),
            "collisions": collisions
        }
        self.results.append(record)
        return record

    def save_report(self):
        """Finalizes all episodes and outputs the final evaluation report (JSON)."""
        if not self.results:
            return None, {}

        total_ep = len(self.results)
        success_rate = sum(1 for r in self.results if r["success"]) / total_ep
        avg_spl = sum(r["spl"] for r in self.results) / total_ep
        avg_time = sum(r["time_taken_s"] for r in self.results) / total_ep
        total_collisions = sum(r["collisions"] for r in self.results)

        summary = {
            "total_episodes": total_ep,
            "success_rate": round(success_rate, 3),
            "average_spl": round(avg_spl, 3),
            "average_time_s": round(avg_time, 2),
            "total_collisions": total_collisions
        }

        report = {
            "summary": summary,
            "episodes": self.results
        }

        file_path = os.path.join(self.output_dir, f"eval_report_{self.timestamp}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)
            
        return file_path, summary