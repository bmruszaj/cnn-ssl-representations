import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any

from src.data.loaders import get_loaders
from src.utils.params_yaml import load_yaml
from src.models.resnet18_with_pools import build_resnet18
from src.models.vae import VAELoss

import argparse

params: Dict[str, Any] = load_yaml()
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

def train_vae_on_pooling(vae_pretrained = True):

    paths = params["paths"]

    ckpt_path = PROJECT_ROOT / paths["checkpoints_dir"] / ("vae_pretrained_model.pt" if vae_pretrained else "vae_model.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_loaders()

    model = build_resnet18(
        pretrained=params["model"]["pretrained"],
        pool_type="vae",
        freeze_pool=False,
        latent=params["model"]["latent_channels"],
        num_classes=params["model"]["num_classes"],
        pretrained_pooling_block=vae_pretrained,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=params['vae']['train']['learning_rate'],
        weight_decay=params['vae']['train']['weight_decay'],
    )
    cls_criterion = nn.CrossEntropyLoss()
    vae_criterion = VAELoss()

    alpha = params['vae']['train']['cls_loss_weight']
    epochs = params['vae']['train']['epochs']

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Run the image through the ResNet backbone up to the last pooling
            x = model.conv1(images)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)

            # Now pass through VAE instead of avgpool
            vae_in = x  # Late features (512 channels)
            recon = model.avgpool(vae_in)  # Using avgpool which contains the VAE

            # Apply adaptive pooling like the original ResNet avgpool would do
            # to get a tensor of shape [batch_size, channels, 1, 1]
            x = torch.nn.functional.adaptive_avg_pool2d(recon, (1, 1))

            # Flatten and pass through fully connected layer
            x = torch.flatten(x, 1)
            logits = model.fc(x)

            vae_l = vae_criterion(model.avgpool, recon, vae_in)
            ce_l = cls_criterion(logits, labels)
            loss = vae_l + alpha * ce_l

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained",
        action="store_true"
    )

    args = parser.parse_args()

    train_vae_on_pooling(args.pretrained)
