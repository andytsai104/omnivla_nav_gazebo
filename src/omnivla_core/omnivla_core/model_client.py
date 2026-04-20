"""Model wrapper / client.

Purpose:
- isolate model loading from ROS node logic
- load base OmniVLA-edge weights + finetuned classifier checkpoint
- expose explicit runtime modes:
    1) prediction
    2) pipeline_check

Behavior:
- prediction:
    preferred path = finetuned classifier inference
    fallback path  = rule-based goal ID prediction
- pipeline_check:
    always use rule-based goal ID prediction
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from goal_library import GoalLibrary
from models.omnivla_edge_classifier_model import OmniVLAEdgeGoalClassifier


class ModelClient:
    """
    Runtime model wrapper.

    Supported runtime modes:
    - prediction:
        use finetuned model if available; fallback to rule-based if needed
    - pipeline_check:
        skip model inference entirely and use rule-based goal matching
    """

    def __init__(
        self,
        model_type: str,
        base_model_checkpoint_path: str,
        classifier_checkpoint_path: str,
        device: str,
        goal_library: GoalLibrary,
    ):
        self.model_type = model_type
        self.base_model_checkpoint_path = str(base_model_checkpoint_path)
        self.classifier_checkpoint_path = str(classifier_checkpoint_path)
        self.goal_library = goal_library

        self.alias_map = goal_library.aliases_to_goal_map(enabled_only=True)
        self.enabled_goals = self.goal_library.get_enabled_goals()

        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )

        self.model: Optional[torch.nn.Module] = None
        self.model_loaded = False
        self.load_error: Optional[str] = None
        self.idx_to_goal_id: List[str] = self._default_idx_to_goal_id()

        self._try_load_finetuned_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_goal_id(
        self,
        image,
        pose_xyyaw,
        prompt: str,
        goal_image=None,
        mode: str = "prediction",
    ) -> tuple[Optional[str], float]:
        """
        Args:
            image:
                Runtime image array. Expected shape HxWx3.
                Can be uint8 [0,255] or float [0,1].
            pose_xyyaw:
                Currently unused by this classifier head.
            prompt:
                Natural language prompt.
            goal_image:
                Unused here, kept for API compatibility.
            mode:
                - "prediction"
                - "pipeline_check"

        Returns:
            (goal_id, confidence)
        """
        if not prompt:
            return None, 0.0

        mode = (mode or "prediction").strip().lower()

        if mode == "pipeline_check":
            return self._predict_goal_id_rule_based(prompt)

        if mode != "prediction":
            return None, 0.0

        if self.model_loaded and self.model is not None:
            try:
                pixel_values = self._runtime_image_to_tensor(image).to(self.device)

                with torch.no_grad():
                    logits = self.model(pixel_values, [prompt])
                    probs = F.softmax(logits, dim=1)
                    conf, pred_idx = torch.max(probs, dim=1)

                idx = int(pred_idx.item())
                confidence = float(conf.item())

                if 0 <= idx < len(self.idx_to_goal_id):
                    goal_id = self.idx_to_goal_id[idx]
                    if goal_id:
                        return goal_id, confidence

                self.load_error = (
                    f"Predicted class index {idx} could not be mapped to a goal_id."
                )

            except Exception as e:
                self.load_error = f"Runtime inference failed: {e}"

        return self._predict_goal_id_rule_based(prompt)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _try_load_finetuned_model(self) -> None:
        base_ckpt = Path(self.base_model_checkpoint_path).expanduser()
        clf_ckpt = Path(self.classifier_checkpoint_path).expanduser()

        if not base_ckpt.exists():
            self.load_error = f"Base model checkpoint not found: {base_ckpt}"
            self.model_loaded = False
            return

        if not clf_ckpt.exists():
            self.load_error = f"Classifier checkpoint not found: {clf_ckpt}"
            self.model_loaded = False
            return

        try:
            checkpoint = torch.load(clf_ckpt, map_location=self.device)

            if not isinstance(checkpoint, dict):
                raise ValueError("Classifier checkpoint is not a dict.")

            num_classes = int(checkpoint.get("num_classes", len(self.enabled_goals)))
            feature_mode = str(checkpoint.get("feature_mode", "actions"))
            hidden_dim = int(checkpoint.get("hidden_dim", 256))
            dropout = float(checkpoint.get("dropout", 0.2))

            self.idx_to_goal_id = self._build_idx_to_goal_id_from_checkpoint(
                checkpoint=checkpoint,
                num_classes=num_classes,
            )

            model = OmniVLAEdgeGoalClassifier(
                checkpoint_path=str(base_ckpt),
                num_classes=num_classes,
                feature_mode=feature_mode,
                device=str(self.device),
                hidden_dim=hidden_dim,
                dropout=dropout,
                unfreeze_model_patterns=[],
                unfreeze_text_encoder=False,
            ).to(self.device)

            state_dict = checkpoint.get("model_state_dict")
            if state_dict is None:
                raise KeyError("Classifier checkpoint missing 'model_state_dict'.")

            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            if missing:
                print(f"[ModelClient] Warning: missing keys: {missing}")
            if unexpected:
                print(f"[ModelClient] Warning: unexpected keys: {unexpected}")

            model.eval()
            self.model = model
            self.model_loaded = True
            self.load_error = None

            print("[ModelClient] Finetuned classifier loaded successfully.")
            print(f"[ModelClient] base_model_checkpoint_path={base_ckpt}")
            print(f"[ModelClient] classifier_checkpoint_path={clf_ckpt}")
            print(f"[ModelClient] idx_to_goal_id={self.idx_to_goal_id}")

        except Exception as e:
            self.model = None
            self.model_loaded = False
            self.load_error = f"Failed to load finetuned model: {e}"
            print(f"[ModelClient] {self.load_error}")
            print("[ModelClient] Falling back to rule-based prediction.")

    # ------------------------------------------------------------------
    # Class-index mapping
    # ------------------------------------------------------------------

    def _default_idx_to_goal_id(self) -> List[str]:
        return [g.goal_id for g in self.enabled_goals]

    def _build_idx_to_goal_id_from_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        num_classes: int,
    ) -> List[str]:
        direct = checkpoint.get("idx_to_goal_id")
        if isinstance(direct, (list, tuple)):
            result = [str(x) for x in direct]
            if len(result) == num_classes:
                return result

        direct_alt = checkpoint.get("class_goal_ids")
        if isinstance(direct_alt, (list, tuple)):
            result = [str(x) for x in direct_alt]
            if len(result) == num_classes:
                return result

        idx_to_label = checkpoint.get("idx_to_label")
        if isinstance(idx_to_label, dict):
            resolved: List[str] = []
            for i in range(num_classes):
                raw = idx_to_label.get(i, idx_to_label.get(str(i)))
                resolved.append(self._resolve_text_to_goal_id(str(raw)) if raw is not None else "")
            if len(resolved) == num_classes and all(resolved):
                return resolved

        if isinstance(idx_to_label, (list, tuple)):
            resolved = [self._resolve_text_to_goal_id(str(x)) for x in idx_to_label]
            if len(resolved) == num_classes and all(resolved):
                return resolved

        fallback = self._default_idx_to_goal_id()
        if len(fallback) < num_classes:
            fallback = fallback + [""] * (num_classes - len(fallback))
        return fallback[:num_classes]

    def _resolve_text_to_goal_id(self, text: str) -> str:
        norm = (text or "").strip().lower()
        if not norm:
            return ""

        if norm in self.alias_map:
            return self.alias_map[norm]

        for alias, goal_id in self.alias_map.items():
            if alias == norm:
                return goal_id

        return ""

    # ------------------------------------------------------------------
    # Image conversion
    # ------------------------------------------------------------------

    def _runtime_image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        if image is None:
            raise ValueError("Input image is None.")

        arr = np.asarray(image)

        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected image shape [H, W, 3], got {arr.shape}")

        arr = arr.astype(np.float32)

        if arr.max() > 1.5:
            arr = arr / 255.0

        arr = np.clip(arr, 0.0, 1.0)

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
        return tensor

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _predict_goal_id_rule_based(self, prompt: str) -> tuple[Optional[str], float]:
        text = prompt.strip().lower()

        if text in self.alias_map:
            return self.alias_map[text], 0.99

        for alias, goal_id in self.alias_map.items():
            if alias in text:
                return goal_id, 0.90

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