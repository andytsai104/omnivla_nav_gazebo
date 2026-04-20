#!/usr/bin/env python3
from __future__ import annotations

"""
PyTorch dataset for goal-ID classification.

Supports two modes:
1) Hugging Face processor mode
   - requires `processor`
   - returns pixel_values + input_ids + attention_mask + labels
2) Image-only debug mode
   - set image_only_debug=True
   - does NOT require `processor`
   - returns pixel_values + labels only (plus metadata)

Expected JSONL schema per line:
{
  "episode_id": "episode_0001",
  "frame_idx": 12,
  "image_path": ".../0012.png",
  "prompt": "go to the big shelf area",
  "goal_id": "big_shelf_area",
  "label_idx": 4,
  "pose": [x, y, yaw],
  "goal_pose": [gx, gy, gyaw]
}
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import torchvision.transforms as T
except ImportError:  # pragma: no cover
    T = None


class GoalClassifierDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        processor=None,
        include_pose: bool = False,
        image_only_debug: bool = False,
        transform=None,
        image_key: str = "image_path",
        prompt_key: str = "prompt",
        label_key: str = "label_idx",
        pose_key: str = "pose",
        max_length: Optional[int] = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")

        self.processor = processor
        self.include_pose = include_pose
        self.image_only_debug = image_only_debug
        self.transform = transform
        self.image_key = image_key
        self.prompt_key = prompt_key
        self.label_key = label_key
        self.pose_key = pose_key
        self.max_length = max_length

        if not self.image_only_debug and self.processor is None:
            raise ValueError(
                "processor must be provided unless image_only_debug=True"
            )

        self.samples = self._load_jsonl(self.jsonl_path)
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {self.jsonl_path}")

        if self.image_only_debug and self.transform is None:
            if T is None:
                raise ImportError(
                    "torchvision is required for image_only_debug mode when no transform is provided"
                )
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {lineno} of {path}: {e}") from e
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, image_path_str: str) -> Path:
        image_path = Path(image_path_str)
        if image_path.is_absolute():
            return image_path
        if image_path.exists():
            return image_path
        candidate = (self.jsonl_path.parent / image_path).resolve()
        return candidate

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        image_path = self._resolve_image_path(sample[self.image_key])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for sample {idx}: {image_path}")

        prompt = str(sample.get(self.prompt_key, ""))
        label_idx = int(sample[self.label_key])
        image = Image.open(image_path).convert("RGB")

        item: Dict[str, Any] = {
            "labels": torch.tensor(label_idx, dtype=torch.long),
            "prompt": prompt,
            "goal_id": sample.get("goal_id", ""),
            "episode_id": sample.get("episode_id", ""),
            "frame_idx": int(sample.get("frame_idx", -1)),
            "image_path": str(image_path),
        }

        if self.image_only_debug:
            item["pixel_values"] = self.transform(image)
        else:
            proc_kwargs = {
                "images": image,
                "text": prompt,
                "return_tensors": "pt",
            }
            if self.max_length is not None:
                proc_kwargs["max_length"] = self.max_length
                proc_kwargs["truncation"] = True

            encoded = self.processor(**proc_kwargs)
            item["pixel_values"] = encoded["pixel_values"].squeeze(0)
            item["input_ids"] = encoded["input_ids"].squeeze(0)
            item["attention_mask"] = encoded["attention_mask"].squeeze(0)

        if self.include_pose:
            pose_vals = sample.get(self.pose_key, [0.0, 0.0, 0.0])
            if len(pose_vals) != 3:
                raise ValueError(
                    f"Expected pose of length 3 for sample {idx}, got {pose_vals}"
                )
            item["pose"] = torch.tensor(pose_vals, dtype=torch.float32)

        return item


class GoalClassifierCollator:
    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(batch) == 0:
            raise ValueError("Empty batch received by GoalClassifierCollator")

        out: Dict[str, Any] = {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch], dim=0),
            "labels": torch.stack([x["labels"] for x in batch], dim=0),
            "prompt": [x["prompt"] for x in batch],
            "goal_id": [x["goal_id"] for x in batch],
            "episode_id": [x["episode_id"] for x in batch],
            "frame_idx": [x["frame_idx"] for x in batch],
            "image_path": [x["image_path"] for x in batch],
        }

        if "input_ids" in batch[0]:
            input_ids = [x["input_ids"] for x in batch]
            attention_masks = [x["attention_mask"] for x in batch]
            out["input_ids"] = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.pad_token_id,
            )
            out["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
                attention_masks,
                batch_first=True,
                padding_value=0,
            )

        if "pose" in batch[0]:
            out["pose"] = torch.stack([x["pose"] for x in batch], dim=0)

        return out
