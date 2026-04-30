#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import csv
import json
import matplotlib.pyplot as plt

from goal_classifier_dataset import GoalClassifierCollator, GoalClassifierDataset
from omnivla_edge_classifier_model import OmniVLAEdgeGoalClassifier


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", type=str, required=True)
    p.add_argument("--val-jsonl", type=str, required=True)
    p.add_argument("--checkpoint-path", type=str, required=True)
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--feature-mode", type=str, default="backbone", choices=["actions", "backbone"])
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--class-weighted-loss", action="store_true")
    p.add_argument("--unfreeze-patterns", type=str, default="transformer,encoder,attention,mha,fusion,goal,obs")
    p.add_argument("--unfreeze-text-encoder", action="store_true")
    p.add_argument("--save-path", type=str, default="omnivla_finetune/checkpoints/test1/omnivla_edge_goal_classifier.pt")
    return p


def compute_class_weights(dataset: GoalClassifierDataset, num_classes: int) -> torch.Tensor:
    counts = Counter(int(s["label_idx"]) for s in dataset.samples)
    weights: List[float] = []
    total = sum(counts.values())
    for c in range(num_classes):
        count = counts.get(c, 1)
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        prompts = batch["prompt"]

        logits = model(pixel_values, prompts)
        loss = criterion(logits, labels)

        total_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}


def save_training_plots(history: List[Dict[str, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = out_dir / "training_history.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
        )
        writer.writeheader()
        writer.writerows(history)

    # Save JSON
    json_path = out_dir / "training_history.json"
    with json_path.open("w") as f:
        json.dump(history, f, indent=2)

    epochs = [h["epoch"] for h in history]

    # Loss plot
    plt.figure()
    plt.plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
    plt.plot(epochs, [h["val_loss"] for h in history], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=300)
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, [h["train_acc"] for h in history], label="Train Accuracy")
    plt.plot(epochs, [h["val_acc"] for h in history], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png", dpi=300)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = GoalClassifierDataset(
        jsonl_path=args.train_jsonl,
        processor=None,
        include_pose=False,
        image_only_debug=True,
    )
    val_ds = GoalClassifierDataset(
        jsonl_path=args.val_jsonl,
        processor=None,
        include_pose=False,
        image_only_debug=True,
    )

    collator = GoalClassifierCollator(pad_token_id=0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)

    sample_batch = next(iter(train_loader))
    print("Sanity check batch:")
    print(f"  labels: {tuple(sample_batch['labels'].shape)}")
    print(f"  goal_id sample: {sample_batch['goal_id'][:2]}")
    print(f"  prompt sample: {sample_batch['prompt'][:2]}")

    unfreeze_patterns = [s.strip() for s in args.unfreeze_patterns.split(",") if s.strip()]
    model = OmniVLAEdgeGoalClassifier(
        checkpoint_path=args.checkpoint_path,
        num_classes=args.num_classes,
        feature_mode=args.feature_mode,
        device=str(device),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        unfreeze_model_patterns=unfreeze_patterns,
        unfreeze_text_encoder=args.unfreeze_text_encoder,
    ).to(device)

    print(f"Trainable parameter count: {model.trainable_parameter_count()}")
    print(f"Unfrozen model params matched patterns: {len(model.extractor.unfrozen_parameter_names)}")

    if args.class_weighted_loss:
        class_weights = compute_class_weights(train_ds, args.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class-weighted loss: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = -1.0
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    plot_dir = save_path.parent / "plots"

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            prompts = batch["prompt"]

            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values, prompts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": args.num_classes,
                    "feature_mode": args.feature_mode,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "unfreeze_patterns": unfreeze_patterns,
                    "best_val_acc": best_val_acc,
                },
                save_path,
            )
            print(f"  Saved new best checkpoint to: {save_path}")

    print(f"Best val_acc: {best_val_acc:.4f}")
    save_training_plots(history, plot_dir)
    print(f"Saved training plots and history to: {plot_dir}")


if __name__ == "__main__":
    main()
