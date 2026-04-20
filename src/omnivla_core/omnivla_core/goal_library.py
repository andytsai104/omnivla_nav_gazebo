"""Goal library helper.

Intended purpose:
- load the semantic goal library YAML
- resolve a goal ID into a pose / metadata
- provide a clean interface for runtime nodes
"""

#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class GoalPose:
    x: float
    y: float
    yaw: float


@dataclass
class GoalEntry:
    goal_id: str
    label: str
    category: str
    frame_id: str
    pose: GoalPose
    goal_image: Optional[str]
    aliases: List[str]
    room: Optional[str]
    enabled: bool
    raw: Dict[str, Any]


class GoalLibrary:
    def __init__(self, yaml_path: str):
        self.yaml_path = Path(yaml_path).expanduser()
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"Goal library not found: {self.yaml_path}")

        with self.yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        goals = data.get("goals", [])
        if not isinstance(goals, list):
            raise ValueError("goal_library.yaml must contain a top-level 'goals' list")

        self._goals_by_id: Dict[str, GoalEntry] = {}
        for item in goals:
            entry = self._parse_goal(item)
            if entry.goal_id in self._goals_by_id:
                raise ValueError(f"Duplicate goal id: {entry.goal_id}")
            self._goals_by_id[entry.goal_id] = entry

    def _parse_goal(self, item: Dict[str, Any]) -> GoalEntry:
        for key in ["id", "label", "frame_id", "pose"]:
            if key not in item:
                raise ValueError(f"Missing required goal field: {key}")

        pose = item["pose"]
        for key in ["x", "y", "yaw"]:
            if key not in pose:
                raise ValueError(f"Missing required pose field: {key}")

        return GoalEntry(
            goal_id=str(item["id"]),
            label=str(item["label"]),
            category=str(item.get("category", "")),
            frame_id=str(item["frame_id"]),
            pose=GoalPose(
                x=float(pose["x"]),
                y=float(pose["y"]),
                yaw=float(pose["yaw"]),
            ),
            goal_image=str(item["goal_image"]) if item.get("goal_image") else None,
            aliases=[str(a) for a in item.get("aliases", [])],
            room=str(item["room"]) if item.get("room") else None,
            enabled=bool(item.get("enabled", True)),
            raw=item,
        )

    def has_goal(self, goal_id: str) -> bool:
        return goal_id in self._goals_by_id

    def get_goal(self, goal_id: str) -> GoalEntry:
        if goal_id not in self._goals_by_id:
            raise KeyError(f"Unknown goal id: {goal_id}")
        return self._goals_by_id[goal_id]

    def get_enabled_goals(self) -> List[GoalEntry]:
        return [g for g in self._goals_by_id.values() if g.enabled]

    def list_goal_ids(self, enabled_only: bool = True) -> List[str]:
        if enabled_only:
            return [g.goal_id for g in self.get_enabled_goals()]
        return list(self._goals_by_id.keys())

    def aliases_to_goal_map(self, enabled_only: bool = True) -> Dict[str, str]:
        result: Dict[str, str] = {}
        goals = self.get_enabled_goals() if enabled_only else self._goals_by_id.values()
        for goal in goals:
            result[goal.goal_id.lower()] = goal.goal_id
            result[goal.label.lower()] = goal.goal_id
            for alias in goal.aliases:
                result[alias.lower()] = goal.goal_id
        return result
