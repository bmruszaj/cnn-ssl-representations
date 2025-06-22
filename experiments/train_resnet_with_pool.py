from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.loaders import get_loaders
from src.utils.params_yaml import load_yaml
from src.models.resnet18_with_pools import build_resnet18

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
params: Dict[str, Any] = load_yaml()


def str2bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def train(pool_type: str, freeze_encoder: bool) -> None:
    tag = f"{pool_type}_{'frozen' if freeze_encoder else 'unfrozen'}"
    ckpt_dir = PROJECT_ROOT / params["paths"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / f"{tag}_model.pt"

    plt_dir = PROJECT_ROOT / params["paths"]["plots_dir"]
    plt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_loaders()

    # Build model with pretrained pool block loaded automatically
    model = build_resnet18(
        pretrained_backbone=False,
        encoder_type=pool_type,
        freeze_encoder=freeze_encoder,
        latent_dim=params["model"]["latent_channels"],
        num_classes=params["model"]["num_classes"],
        pretrained_encoder=True,
    )

    model = model.to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=params[pool_type]["train"]["learning_rate"],
        weight_decay=0.0,
    )
    criterion = nn.CrossEntropyLoss()
    epochs = params[pool_type]["train"]["epochs"]

    train_losses = []
    train_acc = []

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"[Train {tag}] Epoch {epoch}/{epochs}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = logits.argmax(dim=1)
            running_corrects += (preds == labels).sum().item()
            total_samples += labels.size(0)

            avg_loss = running_loss / (pbar.n + 1)
            pbar.set_postfix(loss=avg_loss)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects / total_samples

        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)

    torch.save(model, save_path)
    print(f"Saved full model to {save_path}")


    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, marker='o')
    plt.title(f'pool: {pool_type}, freeze: {freeze_encoder}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plt_dir / f'pool_{pool_type}_freeze_{freeze_encoder}_loss.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_acc, marker='o')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plt_dir / f'pool_{pool_type}_freeze_{freeze_encoder}_acc.png')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool_type", required=True, choices=["ae", "vae"],
        help="Which encoder block to plug into ResNet (ae or vae)."
    )
    parser.add_argument(
        "--freeze_encoder", type=str2bool, default="true", metavar="BOOL",
        help="Freeze the encoder’s parameters during training (true/false)."
    )
    args = parser.parse_args()
    train(args.pool_type, args.freeze_encoder)
