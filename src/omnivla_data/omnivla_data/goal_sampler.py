#!/usr/bin/env python3
"""
goal_sampler.py
---------------
Intelligently samples start and goal poses from the goal library.
Ensures uniform distribution of goals by tracking visit history,
and guarantees minimum distance constraints between start and goal.
"""

import math
import random

class GoalSampler:
    def __init__(self, goals: list, min_dist: float = 1.5):
        """
        :param goals: List of goal dictionaries loaded from goal_library.yaml
        :param min_dist: Minimum Euclidean distance between start and goal
        """
        self.goals = goals
        self.min_dist = min_dist
        
        # Track how many times each goal has been used as a target
        self.history = {g['id']: 0 for g in self.goals}

    def _euclidean_dist(self, p1: dict, p2: dict) -> float:
        return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

    def sample_target(self) -> dict:
        """
        Samples a target goal, prioritizing goals that have been visited the least.
        """
        if not self.goals:
            raise ValueError("Goal list is empty.")
            
        # Find the minimum visit count
        min_visits = min(self.history.values())
        
        # Get all goals that have this minimum count
        candidates = [g for g in self.goals if self.history[g['id']] == min_visits]
        
        # Randomly pick from the least-visited candidates
        chosen = random.choice(candidates)
        
        return chosen

    def sample_start(self, target_goal: dict) -> dict:
        """
        Samples a start pose that is at least `min_dist` away from the target goal.
        Returns None if no valid start pose is found.
        """
        if len(self.goals) < 2:
            return None
            
        candidates = [
            g for g in self.goals
            if g['id'] != target_goal['id'] 
            and self._euclidean_dist(g['pose'], target_goal['pose']) >= self.min_dist
        ]
        
        if not candidates:
            return None
            
        return random.choice(candidates)

    def record_visit(self, goal_id: str):
        """
        Increments the visit count for a given goal. Must be called after a successful episode.
        """
        if goal_id in self.history:
            self.history[goal_id] += 1
