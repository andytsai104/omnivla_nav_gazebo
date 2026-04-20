#!/usr/bin/env python3
"""
Export raw OmniVLA navigation episodes into JSONL splits for goal-ID classification.

Input layout (from data_logger_node.py):
  <run_dir>/
    episode_0001/
      metadata.json
      frames/
        0000.png
        0000.json
        ...

Output:
  <out_dir>/
    train.jsonl
    val.jsonl
    test.jsonl
    label_map.json
    split_summary.json

Each JSONL line contains:
{
  "episode_id": "episode_0001",
  "frame_idx": 12,
  "image_path": "/abs/or/relative/path/to/0012.png",
  "prompt": "go to the big shelf area",
  "goal_id": "big_shelf_area",
  "label_idx": 4,
  "pose": [x, y, yaw],
  "goal_pose": [gx, gy, gyaw]
}
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_goal_ids(goal_library_path: Path) -> List[str]:
    with goal_library_path.open("r", encoding="utf-8") as f:
        data = json.loads(json.dumps(__import__('yaml').safe_load(f)))
    goals = data.get("goals", [])
    enabled = [g for g in goals if g.get("enabled", True)]
    goal_ids = [str(g["id"]) for g in enabled]
    if not goal_ids:
        raise ValueError(f"No enabled goals found in {goal_library_path}")
    return goal_ids


def count_frame_jsons(frames_dir: Path) -> int:
    return len(list(frames_dir.glob("*.json")))


def select_frame_jsons(frames_dir: Path, keep_every_n: int) -> List[Path]:
    frame_jsons = sorted(frames_dir.glob("*.json"))
    if keep_every_n <= 1:
        return frame_jsons
    return [p for i, p in enumerate(frame_jsons) if i % keep_every_n == 0]


def split_episode_ids(
    episode_ids: List[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    ids = list(episode_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_test = max(1 if n >= 5 and test_ratio > 0 else 0, int(round(n * test_ratio)))
    n_val = max(1 if n >= 5 and val_ratio > 0 else 0, int(round(n * val_ratio)))

    if n_test + n_val >= n:
        n_test = min(n_test, max(0, n - 2))
        n_val = min(n_val, max(0, n - n_test - 1))

    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test + n_val]
    train_ids = ids[n_test + n_val:]
    return train_ids, val_ids, test_ids


def build_sample(frame_json_path: Path, image_path: Path, metadata: dict, label_idx: int, image_path_mode: str) -> dict:
    with frame_json_path.open("r", encoding="utf-8") as f:
        frame = json.load(f)

    odom = frame.get("odom", {})
    pos = odom.get("position", {})
    pose = [
        float(pos.get("x", 0.0)),
        float(pos.get("y", 0.0)),
        float(odom.get("yaw_rad", 0.0)),
    ]

    gp = frame.get("goal_pose") or metadata.get("goal_pose") or {}
    goal_pose = [
        float(gp.get("x", 0.0)),
        float(gp.get("y", 0.0)),
        float(gp.get("yaw_rad", 0.0)),
    ]

    if image_path_mode == "absolute":
        image_path_str = str(image_path.resolve())
    else:
        image_path_str = str(image_path)

    return {
        "episode_id": metadata.get("episode_id", frame_json_path.parent.parent.name),
        "frame_idx": int(frame.get("frame_idx", -1)),
        "image_path": image_path_str,
        "prompt": frame.get("prompt") or metadata.get("prompt", ""),
        "goal_id": frame.get("goal_id") or metadata.get("goal_id", ""),
        "label_idx": int(label_idx),
        "pose": pose,
        "goal_pose": goal_pose,
    }


def write_jsonl(path: Path, samples: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export goal-classification JSONL splits from raw episode folders.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Raw dataset run directory, e.g. ./datasets/run_001")
    parser.add_argument("--goal-library", type=Path, required=True, help="Path to goal_library.yaml")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for JSONL splits")
    parser.add_argument("--keep-every-n", type=int, default=4, help="Keep every Nth frame JSON within each episode")
    parser.add_argument("--success-only", action="store_true", help="Only export episodes with outcome='success'")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-path-mode", choices=["relative", "absolute"], default="relative")
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    goal_ids = sorted(load_goal_ids(args.goal_library))
    label_map = {goal_id: idx for idx, goal_id in enumerate(goal_ids)}

    episode_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")])
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_* directories found in {run_dir}")

    episodes_by_goal: Dict[str, List[Tuple[str, Path, dict]]] = defaultdict(list)
    skipped = []

    for ep_dir in episode_dirs:
        meta_path = ep_dir / "metadata.json"
        frames_dir = ep_dir / "frames"
        if not meta_path.exists() or not frames_dir.exists():
            skipped.append({"episode_id": ep_dir.name, "reason": "missing metadata.json or frames/"})
            continue

        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if args.success_only and metadata.get("outcome") != "success":
            skipped.append({"episode_id": ep_dir.name, "reason": f"outcome={metadata.get('outcome')}"})
            continue

        goal_id = metadata.get("goal_id", "")
        if goal_id not in label_map:
            skipped.append({"episode_id": ep_dir.name, "reason": f"unknown goal_id={goal_id}"})
            continue

        if count_frame_jsons(frames_dir) == 0:
            skipped.append({"episode_id": ep_dir.name, "reason": "no frame json files"})
            continue

        episodes_by_goal[goal_id].append((ep_dir.name, ep_dir, metadata))

    split_to_samples = {"train": [], "val": [], "test": []}
    split_to_episode_ids = {"train": [], "val": [], "test": []}

    for goal_id, ep_items in sorted(episodes_by_goal.items()):
        episode_ids = [ep_id for ep_id, _, _ in ep_items]
        train_ids, val_ids, test_ids = split_episode_ids(
            episode_ids,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        membership = {ep_id: "train" for ep_id in train_ids}
        membership.update({ep_id: "val" for ep_id in val_ids})
        membership.update({ep_id: "test" for ep_id in test_ids})

        for ep_id, ep_dir, metadata in ep_items:
            split = membership[ep_id]
            split_to_episode_ids[split].append(ep_id)
            frame_jsons = select_frame_jsons(ep_dir / "frames", args.keep_every_n)

            for frame_json_path in frame_jsons:
                frame_idx = int(frame_json_path.stem)
                image_path = ep_dir / "frames" / f"{frame_idx:04d}.png"
                if not image_path.exists():
                    continue
                sample = build_sample(
                    frame_json_path=frame_json_path,
                    image_path=image_path,
                    metadata=metadata,
                    label_idx=label_map[goal_id],
                    image_path_mode=args.image_path_mode,
                )
                split_to_samples[split].append(sample)

    for split in ["train", "val", "test"]:
        write_jsonl(out_dir / f"{split}.jsonl", split_to_samples[split])

    with (out_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    summary = {
        "run_dir": str(run_dir),
        "goal_library": str(args.goal_library),
        "keep_every_n": args.keep_every_n,
        "success_only": args.success_only,
        "label_map": label_map,
        "episodes_per_goal": {goal_id: len(items) for goal_id, items in episodes_by_goal.items()},
        "split_episode_counts": {k: len(v) for k, v in split_to_episode_ids.items()},
        "split_sample_counts": {k: len(v) for k, v in split_to_samples.items()},
        "split_goal_counts": {
            split: dict(Counter(sample["goal_id"] for sample in samples))
            for split, samples in split_to_samples.items()
        },
        "skipped": skipped,
    }
    with (out_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Export complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
