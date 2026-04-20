#!/usr/bin/env python3
from __future__ import annotations

"""
Tiny debug training script for goal-ID classification.

Purpose:
- verify exported JSONL loads correctly
- verify labels are sane
- verify a simple classifier can overfit a tiny subset

This intentionally uses an image-only debug path first.
If this script can overfit a tiny subset, your pipeline is wired correctly.
"""

import argparse
import random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from goal_classifier_dataset import GoalClassifierDataset, GoalClassifierCollator


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x).flatten(1)
        return self.classifier(feats)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total = 0

        for batch in loader:
            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / max(total, 1), total_correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, default="")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tiny-subset", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds_full = GoalClassifierDataset(
        jsonl_path=args.train_jsonl,
        processor=None,
        include_pose=False,
        image_only_debug=True,
    )

    subset_size = min(args.tiny_subset, len(train_ds_full))
    subset_indices = list(range(subset_size))
    train_ds = Subset(train_ds_full, subset_indices)

    collator = GoalClassifierCollator(pad_token_id=0)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, subset_size),
        shuffle=True,
        collate_fn=collator,
    )

    val_loader = None
    if args.val_jsonl:
        val_ds = GoalClassifierDataset(
            jsonl_path=args.val_jsonl,
            processor=None,
            include_pose=False,
            image_only_debug=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

    sample_batch = next(iter(train_loader))
    print("Sanity check batch:")
    print("  pixel_values:", tuple(sample_batch["pixel_values"].shape))
    print("  labels:", tuple(sample_batch["labels"].shape))
    print("  goal_id sample:", sample_batch["goal_id"][:2])
    print("  prompt sample:", sample_batch["prompt"][:2])

    model = TinyClassifier(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch in train_loader:
            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)

        train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)

        msg = f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f}"

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, device)
            msg += f" | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"

        print(msg)

    print("\nDone.")
    print("If train_acc gets very high on the tiny subset, your classifier pipeline is working.")


if __name__ == "__main__":
    main()
