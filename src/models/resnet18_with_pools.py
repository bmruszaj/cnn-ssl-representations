# resnet18_with_pools.py
"""Build ResNet‑18 with a pluggable replacement for the final AvgPool2d layer.

Supported `pool_type` values:
    "max"   – leave the original MaxPool2d
    "ae"    – deterministic Autoencoder block (AEPool)
    "ae_pretrained" – AE block with pretrained weights
    "vae"   – Variational‑AE block  (VAEPool)
    "vae_pretrained" – VAE block with pretrained weights
    "simclr"– placeholder for a SimCLR projection/encoder block
    "byol"  – placeholder for a BYOL projection/encoder block
"""

from typing import Literal, Dict, Any
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from src.models.vae import VAE
from src.models.ae import Autoencoder as AE
from src.utils.params_yaml import load_yaml
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
cfg: Dict[str, Any] = load_yaml()


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def build_resnet18(
    *,
    pretrained: bool,
    pool_type: Literal["max", "ae", "ae_pretrained", "vae", "vae_pretrained", "simclr", "byol"] = "max",
    freeze_pool: bool = False,
    latent: int = 128,
    num_classes: int = 10,
    pretrained_pooling_block: bool = False,
) -> nn.Module:
    """Return a ResNet‑18 variant with a chosen pooling/projection block."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    if pool_type != "max":
        # For AE we use early features (64 channels from conv1)
        # For VAE we use late features (512 channels from layer4)
        if pool_type in ("ae", "ae_pretrained"):
            in_ch = model.conv1.out_channels  # 64 channels for early features
        else:
            in_ch = 512  # 512 channels from layer4 output for VAE and other models

        if pool_type in ("ae", "ae_pretrained"):
            block: nn.Module = AE(in_ch, latent)
            if pool_type == "ae_pretrained" or pretrained_pooling_block:
                path = PROJECT_ROOT / cfg['paths']["pretrain_dir"] / "ae_pretrained.pth"
                if path.exists():
                    block.load_state_dict(torch.load(path, map_location="cpu"))
                else:
                    raise FileNotFoundError(f"Pretrained AE not found at {path!r}")

        elif pool_type in ("vae", "vae_pretrained"):
            block = VAE(
                in_ch,
                cfg['vae']['hidden_ch'],
                cfg['vae']['latent_ch'],
                cfg['vae']['beta']
            )
            if pool_type == "vae_pretrained" or pretrained_pooling_block:
                path = PROJECT_ROOT / cfg['paths']["pretrain_dir"] / "vae_pretrained.pth"
                if path.exists():
                    block.load_state_dict(torch.load(path, map_location="cpu"))
                else:
                    raise FileNotFoundError(f"Pretrained VAE not found at {path!r}")

        # other stubs
        elif pool_type == "simclr":
            from src.models.resnet18_with_pools import SimCLRPool
            block = SimCLRPool(in_ch)
        elif pool_type == "byol":
            from src.models.resnet18_with_pools import BYOLPool
            block = BYOLPool(in_ch)
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type!r}")

        model.avgpool = block
        if freeze_pool:
            _freeze(block)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Stubs for other pools (placed below or imported as needed)
class SimCLRPool(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class BYOLPool(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
