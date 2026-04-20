"""Model wrapper / client.

Intended purpose:
- isolate OmniVLA-side imports from the ROS node logic
- load model + checkpoint
- expose a simple predict(...) API
"""

#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

from .goal_library import GoalLibrary


class ModelClient:
    """
    Temporary runtime model wrapper for testing inference_nav.launch.py.

    Current behavior:
    - rule-based goal ID prediction using prompt text
    - lets you validate:
      prompt -> inferred_goal_id -> nav2 goal bridge -> Nav2 motion

    Later you can replace predict_goal_id() with real OmniVLA-edge inference.
    """

    def __init__(
        self,
        model_type: str,
        checkpoint_path: str,
        device: str,
        goal_library: GoalLibrary,
    ):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.goal_library = goal_library
        self.alias_map = goal_library.aliases_to_goal_map(enabled_only=True)

    def predict_goal_id(
        self,
        image,
        pose_xyyaw,
        prompt: str,
        goal_image=None,
    ) -> tuple[Optional[str], float]:
        if not prompt:
            return None, 0.0

        text = prompt.strip().lower()

        # exact alias / label / id match
        if text in self.alias_map:
            return self.alias_map[text], 0.99

        # substring match
        for alias, goal_id in self.alias_map.items():
            if alias in text:
                return goal_id, 0.90

        # keyword overlap fallback
        prompt_tokens = set(text.replace("_", " ").split())
        best_goal = None
        best_score = 0.0
        for goal in self.goal_library.get_enabled_goals():
            candidates = [goal.goal_id, goal.label] + goal.aliases
            for c in candidates:
                tokens = set(c.lower().replace("_", " ").split())
                overlap = len(prompt_tokens & tokens)
                if overlap > best_score:
                    best_score = float(overlap)
                    best_goal = goal.goal_id

        if best_goal is None or best_score <= 0.0:
            return None, 0.0

        conf = min(0.8, 0.3 + 0.1 * best_score)
        return best_goal, conf
