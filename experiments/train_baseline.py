import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any

from src.data.loaders import get_loaders
from src.utils.params_yaml import load_yaml
from src.models.resnet18_with_pools import build_resnet18

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
params: Dict[str, Any] = load_yaml()


def train_baseline():
    train_cfg = params["baseline"]["train"]
    paths = params["paths"]

    epochs = train_cfg["epochs"]
    lr = train_cfg["learning_rate"]

    ckpt_path = PROJECT_ROOT / paths["checkpoints_dir"] / "baseline_model.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_loaders()

    model = build_resnet18(
        pretrained=params["model"]["pretrained"],
        pool_type="max",
        freeze_pool=False,
        latent=params["model"]["latent_channels"],
        num_classes=params["model"]["num_classes"],
    )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    train_baseline()
