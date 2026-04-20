#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T

_WS = Path(__file__).resolve().parents[1]
_CANDIDATES = [
    _WS / "OmniVLA",
    Path.home() / "ros2_projects" / "omnivla_ws" / "OmniVLA",
    Path.home() / "OmniVLA",
]
for p in _CANDIDATES:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import clip  # type: ignore
from utils_policy import load_model  # type: ignore


def default_image_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])


class OmniVLAEdgeFeatureExtractor(nn.Module):
    """
    Feature extractor wrapper around OmniVLA-edge.

    Supported feature modes:
      - "actions": flatten final predicted actions. This is always supported.
      - "backbone": try to extract an internal hidden feature from the model output.
                    If no suitable hidden feature is exposed by the model, safely
                    fall back to action features instead of causing a shape mismatch.

    The key design goal here is that `output_feature_dim` must always match what
    `forward_features()` actually returns.
    """

    def __init__(
        self,
        checkpoint_path: str,
        feature_mode: str = "actions",
        device: str = "cuda",
        unfreeze_model_patterns: Optional[List[str]] = None,
        unfreeze_text_encoder: bool = False,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.requested_feature_mode = feature_mode
        self.feature_mode = feature_mode
        self.transform = default_image_transform(image_size)

        model_params: Dict[str, object] = {
            "model_type": "omnivla-edge",
            "len_traj_pred": 8,
            "learn_angle": True,
            "context_size": 5,
            "obs_encoder": "efficientnet-b0",
            "encoding_size": 256,
            "obs_encoding_size": 1024,
            "goal_encoding_size": 1024,
            "late_fusion": False,
            "mha_num_attention_heads": 4,
            "mha_num_attention_layers": 4,
            "mha_ff_dim_factor": 4,
            "clip_type": "ViT-B/32",
        }

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.model_params = model_params
        self.edge_model, self.text_encoder, _ = load_model(checkpoint_path, model_params, self.device)
        self.edge_model = self.edge_model.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)

        for p in self.edge_model.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = bool(unfreeze_text_encoder)

        self.unfrozen_parameter_names: List[str] = []
        if unfreeze_model_patterns:
            self.unfrozen_parameter_names = self._unfreeze_by_patterns(self.edge_model, unfreeze_model_patterns)

        self.action_dim = int(model_params["len_traj_pred"]) * 4
        self.backbone_dim = self._infer_declared_backbone_dim()
        self.output_feature_dim = self._resolve_output_feature_dim()

    def _unfreeze_by_patterns(self, module: nn.Module, patterns: Iterable[str]) -> List[str]:
        patterns_l = [p.lower() for p in patterns if p]
        unfrozen: List[str] = []
        for name, param in module.named_parameters():
            lname = name.lower()
            if any(p in lname for p in patterns_l):
                param.requires_grad = True
                unfrozen.append(name)
        return unfrozen

    def _infer_declared_backbone_dim(self) -> int:
        for name in ["encoding_size", "obs_encoding_size", "goal_encoding_size", "hidden_dim", "embed_dim", "feature_dim"]:
            if hasattr(self.edge_model, name):
                try:
                    value = int(getattr(self.edge_model, name))
                    if value > 0:
                        return value
                except Exception:
                    pass
        return self.action_dim

    def _resolve_output_feature_dim(self) -> int:
        if self.feature_mode == "actions":
            return self.action_dim

        if self.feature_mode == "backbone":
            # We only know the true backbone dimension if the model actually exposes
            # a hidden feature tensor. Since that is model-dependent and may not be
            # available, make the contract safe by defaulting to action_dim.
            #
            # If a real hidden feature is discovered at runtime, forward_features()
            # will return it only if its dimensionality matches `backbone_dim`.
            # Otherwise we fall back to action features.
            return self.action_dim

        raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _images_to_edge_input(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values.repeat(1, 6, 1, 1).to(self.device)

    def _goal_pose_dummy(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, 4, device=self.device)

    def _map_images_dummy(self, pixel_values: torch.Tensor) -> torch.Tensor:
        b, _, h, w = pixel_values.shape
        zeros = torch.zeros(b, 6, h, w, device=self.device, dtype=pixel_values.dtype)
        return torch.cat([zeros, pixel_values.to(self.device)], dim=1)

    def _goal_image_dummy(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values.to(self.device)

    def _text_features(self, prompts: List[str]) -> torch.Tensor:
        tokens = clip.tokenize(prompts, truncate=True).to(self.device)
        with torch.set_grad_enabled(any(p.requires_grad for p in self.text_encoder.parameters())):
            text_feat = self.text_encoder.encode_text(tokens)
        return text_feat

    def _predict_raw_output(self, pixel_values: torch.Tensor, prompts: List[str]):
        batch_size = pixel_values.shape[0]
        obs_images = self._images_to_edge_input(pixel_values)
        goal_pose = self._goal_pose_dummy(batch_size)
        map_images = self._map_images_dummy(pixel_values)
        goal_image = self._goal_image_dummy(pixel_values)
        cur_large_img = pixel_values.to(self.device)
        text_feat = self._text_features(prompts)
        modality_id = torch.full((batch_size,), 7, dtype=torch.long, device=self.device)

        out = self.edge_model(
            obs_images,
            goal_pose,
            map_images,
            goal_image,
            modality_id,
            text_feat,
            cur_large_img,
        )
        return out, modality_id

    @staticmethod
    def _flatten_feature(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x.reshape(x.shape[0], -1)

    def _extract_action_feature(self, raw_out) -> torch.Tensor:
        predicted_actions = raw_out[0] if isinstance(raw_out, tuple) else raw_out
        if not isinstance(predicted_actions, torch.Tensor):
            raise TypeError("Model output does not contain tensor predicted actions in position 0.")
        return self._flatten_feature(predicted_actions)

    def _candidate_hidden_tensors(self, raw_out) -> Sequence[torch.Tensor]:
        """
        Collect tensor candidates from the model output that might represent hidden
        features instead of final action predictions.
        """
        candidates: List[torch.Tensor] = []

        if isinstance(raw_out, torch.Tensor):
            return candidates

        if isinstance(raw_out, (tuple, list)):
            for idx, item in enumerate(raw_out[1:], start=1):
                if isinstance(item, torch.Tensor):
                    candidates.append(item)
                elif isinstance(item, (tuple, list)):
                    for nested in item:
                        if isinstance(nested, torch.Tensor):
                            candidates.append(nested)
                elif isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, torch.Tensor):
                            candidates.append(v)

        elif isinstance(raw_out, dict):
            for v in raw_out.values():
                if isinstance(v, torch.Tensor):
                    candidates.append(v)

        return candidates

    def _extract_backbone_feature(self, raw_out) -> Tuple[torch.Tensor, bool]:
        """
        Try to recover a real hidden feature from the raw model output.

        Returns:
          feats: [B, D]
          is_true_backbone: whether this came from an internal hidden tensor rather
                            than falling back to action output.
        """
        action_feats = self._extract_action_feature(raw_out)
        batch_size = action_feats.shape[0]

        # Prefer tensors whose flattened size matches the model-declared backbone dim.
        for tensor in self._candidate_hidden_tensors(raw_out):
            if tensor.shape[0] != batch_size:
                continue
            flat = self._flatten_feature(tensor)
            if flat.shape[1] == self.backbone_dim and flat.shape[1] != self.action_dim:
                return flat, True

        # Secondary heuristic: choose the largest non-action tensor if it provides a
        # richer representation and is consistent across batch.
        best_flat: Optional[torch.Tensor] = None
        for tensor in self._candidate_hidden_tensors(raw_out):
            if tensor.shape[0] != batch_size:
                continue
            flat = self._flatten_feature(tensor)
            if flat.shape[1] > self.action_dim:
                if best_flat is None or flat.shape[1] > best_flat.shape[1]:
                    best_flat = flat

        if best_flat is not None:
            return best_flat, True

        # Safe fallback.
        return action_feats, False

    def forward_features(self, pixel_values: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        raw_out, _ = self._predict_raw_output(pixel_values, prompts)

        if self.feature_mode == "actions":
            feats = self._extract_action_feature(raw_out)
            if feats.shape[1] != self.output_feature_dim:
                raise RuntimeError(
                    f"Action feature dim mismatch: got {feats.shape[1]}, expected {self.output_feature_dim}."
                )
            return feats

        if self.feature_mode == "backbone":
            feats, is_true_backbone = self._extract_backbone_feature(raw_out)

            if not is_true_backbone and self.requested_feature_mode == "backbone":
                warnings.warn(
                    "feature_mode='backbone' was requested, but no internal backbone feature "
                    "tensor was exposed by the OmniVLA-edge model output. Falling back to "
                    "flattened action features. Training will still run, but this is not a "
                    "true backbone embedding.",
                    stacklevel=2,
                )

            if feats.shape[1] != self.output_feature_dim:
                # Keep the contract safe even if a true hidden feature is discovered.
                # If you later decide to use real backbone features, update both this
                # extractor and the classifier head initialization together.
                warnings.warn(
                    f"Resolved feature dim {feats.shape[1]} does not match configured "
                    f"output_feature_dim {self.output_feature_dim}. Falling back to action features.",
                    stacklevel=2,
                )
                feats = self._extract_action_feature(raw_out)

            return feats

        raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")


class OmniVLAEdgeGoalClassifier(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int,
        feature_mode: str = "actions",
        device: str = "cuda",
        hidden_dim: int = 256,
        dropout: float = 0.2,
        unfreeze_model_patterns: Optional[List[str]] = None,
        unfreeze_text_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.extractor = OmniVLAEdgeFeatureExtractor(
            checkpoint_path=checkpoint_path,
            feature_mode=feature_mode,
            device=device,
            unfreeze_model_patterns=unfreeze_model_patterns,
            unfreeze_text_encoder=unfreeze_text_encoder,
        )
        feat_dim = self.extractor.output_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        feats = self.extractor.forward_features(pixel_values, prompts)
        return self.classifier(feats)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)