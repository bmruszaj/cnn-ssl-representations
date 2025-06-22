from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.optim as optim
from tqdm import tqdm
from src.models.resnet18_with_pools import build_resnet18

from src.data.loaders import get_loaders
from src.utils.params_yaml import load_yaml
from src.models.ae import Autoencoder as AE, AELoss
from src.models.vae import VAE, VAELoss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
params: Dict[str, Any] = load_yaml()


def str2pool(v: str) -> str:
    v = v.lower()
    if v not in {"ae", "vae"}:
        raise argparse.ArgumentTypeError("pool_type must be 'ae' or 'vae'.")
    return v


# ---------------------------------------------------------------------------


def pretrain(pool_type: str) -> None:
    cfg = params[pool_type]  # ae.* or vae.* sub‑tree
    pre_cfg = cfg["pretrain"]  # epochs, lr, etc.
    latent_c = params["model"]["latent_channels"]

    model = build_resnet18(
        pretrained_backbone=params["model"]["pretrained"],
        pool_type=pool_type,
        freeze_encoder=False,
        latent_dim=latent_c,
        num_classes=params["model"]["num_classes"],
        pretrained_pooling_block=False,
    )
    for p in model.features.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        feat_ch = model.features(torch.zeros(1, 3, 224, 224)).shape[1]
    if pool_type == "ae":
        pool_block = AE(in_channels=feat_ch, latent_channels=latent_c)
    else:
        pool_block = VAE(
            in_ch=feat_ch,
            hidden_ch=params["vae"]["hidden_ch"],
            latent_dim=latent_c,
            beta=params["vae"]["beta"],
        )

    if pool_type == "ae":
        criterion = AELoss(recon_loss_weight=params["ae"]["recon_loss_weight"])
    else:
        criterion = VAELoss(beta=params["vae"]["beta"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    pool_block = pool_block.to(device)

    optimizer = optim.AdamW(
        pool_block.parameters(),
        lr=pre_cfg["learning_rate"],
        weight_decay=0.0,
    )
    train_loader, _ = get_loaders()

    epochs = pre_cfg["epochs"]
    for epoch in range(1, epochs + 1):
        pool_block.train()
        running = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"[Pretrain {pool_type.upper()}] Epoch {epoch}/{epochs}",
            leave=False,
        )
        for imgs, _ in pbar:
            imgs = imgs.to(device)

            with torch.no_grad():
                feats = model.features(imgs)

            if pool_type == "ae":
                recon, _ = pool_block(feats)
                if recon.shape != feats.shape:
                    recon = recon[:, :, : feats.size(2), : feats.size(3)]
                loss = criterion(recon, feats)

            else:
                recon = pool_block(feats)
                if recon.shape != feats.shape:
                    recon = recon[:, :, : feats.size(2), : feats.size(3)]
                loss = criterion(pool_block, recon, feats)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            pbar.set_postfix(loss=running / (pbar.n + 1))

    ckpt_dir = PROJECT_ROOT / params["paths"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / f"{pool_type}_pretrained.pth"
    torch.save(pool_block.state_dict(), save_path)
    print(f"Saved pretrained {pool_type.upper()} block to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_type", type=str2pool, required=True, help="ae or vae")
    args = parser.parse_args()
    pretrain(args.pool_type)
