# resnet18_with_pools.py
"""Build ResNet‑18 with a pluggable replacement for the first MaxPool2d layer.

Supported `pool_type` values:
    "max"   – leave the original MaxPool2d
    "ae"    – deterministic Autoencoder block (AEPool)
    "vae"   – Variational‑AE block  (VAEPool)
    "simclr"– placeholder for a SimCLR projection/encoder block
    "byol"  – placeholder for a BYOL projection/encoder block

The SimCLR / BYOL blocks below are *stubs* so that teammates can later
fill in real implementations without touching the builder signature.
"""

from typing import Literal
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class Autoencoder(nn.Module):
    "Stub that mimics pooling semantics for Autoencoder experiments."

    def __init__(self, in_channels: int, latent: int):
        super().__init__()
        self.proj = nn.Identity()  # TODO: replace with real Autoencoder projection

    def forward(self, x):  # type: ignore[override]
        return self.proj(x)


class VariationalAutoencoder(nn.Module):
    """Stub that mimics pooling semantics for Variational Autoencoder experiments."""

    def __init__(self, in_channels: int, latent: int):
        super().__init__()
        self.proj = nn.Identity()  # TODO: replace with real VAE projection

    def forward(self, x):  # type: ignore[override]
        return self.proj(x)


class SimCLRPool(nn.Module):
    """Stub that mimics pooling semantics for SimCLR experiments."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Identity()  # TODO: replace with real SimCLR projection

    def forward(self, x):  # type: ignore[override]
        return self.proj(x)


class BYOLPool(nn.Module):
    """Stub that mimics pooling semantics for BYOL experiments."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Identity()  # TODO: replace with real BYOL projection

    def forward(self, x):  # type: ignore[override]
        return self.proj(x)


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def build_resnet18(
    *,
    pretrained: bool,
    pool_type: Literal["max", "ae", "vae", "simclr", "byol"] = "max",
    freeze_pool: bool = False,
    latent: int = 128,
    num_classes: int = 10,
) -> nn.Module:
    """Return a ResNet‑18 variant with a chosen pooling/projection block.

    Parameters
    ----------
    pretrained : bool
        If *True*, ImageNet weights are loaded.
    pool_type : str
        "max" (default) keeps MaxPool2d; other options insert learnable blocks.
    freeze_pool : bool
        If *True*, inserted block parameters will be frozen.
    latent : int
        Latent channels for AE/VAE blocks.
    num_classes : int
        Output classes for the final classifier head.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    if pool_type != "max":
        in_ch = model.conv1.out_channels
        if pool_type == "ae":
            block: nn.Module = Autoencoder(in_ch, latent)
        elif pool_type == "vae":
            block = VariationalAutoencoder(in_ch, latent)
        elif pool_type == "simclr":
            block = SimCLRPool(in_ch)
        elif pool_type == "byol":
            block = BYOLPool(in_ch)
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}")

        model.maxpool = block
        if freeze_pool:
            _freeze(block)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
