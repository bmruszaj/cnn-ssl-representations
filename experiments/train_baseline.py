import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any

from torchvision.models import resnet18, ResNet18_Weights

from src.data.loaders import get_loaders
from src.utils.params_yaml import load_yaml

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

    # Build plain ResNet-18
    weights = ResNet18_Weights.IMAGENET1K_V1 if params["model"]["pretrained"] else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, params["model"]["num_classes"])
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
    print(f"✅ Model saved to {ckpt_path}")


if __name__ == "__main__":
    train_baseline()
