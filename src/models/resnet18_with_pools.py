from __future__ import annotations
from typing import Literal, Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from src.utils.params_yaml import load_yaml
from src.models.ae import Autoencoder as AE
from src.models.vae import VAE

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
cfg: Dict[str, Any] = load_yaml()


# --------------------------------------------------------------------------- helpers
def _freeze(m: nn.Module):  # freeze util
    for p in m.parameters():
        p.requires_grad_(False)


class _VAEEncoder(nn.Module):
    """Wrap VAE to output reparameterized latent tensor of shape [B, latent, H, W]"""

    def __init__(self, vae: VAE):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.vae.encoder(x)
        return self.vae.reparameterize(mu, logvar)


# --------------------------------------------------------------------------- backbone-encoder-head
class ResNetWithEncoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        latent_dim: int,
        num_classes: int,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # conv→layer4
        # set encoder (pooling block)
        self.encoder = encoder
        self.maxpool = encoder  # alias for backward compatibility
        if freeze_encoder:
            _freeze(self.encoder)
        # classifier head
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.fc = self.classifier  # alias for training scripts

    def _get_latent(self, enc_out):
        """AE returns (recon, z), VAE returns z directly."""
        if isinstance(enc_out, tuple):
            return enc_out[-1]  # take last element
        return enc_out

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        x = self.features(x)  # (B,512,7,7)
        z = self._get_latent(self.encoder(x))  # (B,latent_dim) or 4-D
        z = torch.flatten(z, 1)  # ensure 2-D
        return self.classifier(z)


# --------------------------------------------------------------------------- factory
EncoderType = Literal["ae", "vae"]


def _build_encoder(
    enc_type: EncoderType, flat_dim: int, latent: int, pretrained: bool
) -> nn.Module:
    if enc_type == "ae":
        ae = AE(in_channels=512, latent_channels=latent)
        if pretrained:
            ae.load_state_dict(
                torch.load(
                    PROJECT_ROOT
                    / cfg["paths"]["checkpoints_dir"]
                    / "ae_pretrained.pth",
                    map_location="cpu",
                ),
                strict=False,
            )
        return ae.encoder  # conv encoder

    if enc_type == "vae":
        vae = VAE(
            in_ch=flat_dim,
            hidden_ch=cfg["vae"]["hidden_ch"],
            latent_dim=latent,
            beta=cfg["vae"]["beta"],
        )
        if pretrained:
            vae.load_state_dict(
                torch.load(
                    PROJECT_ROOT
                    / cfg["paths"]["checkpoints_dir"]
                    / "vae_pretrained.pth",
                    map_location="cpu",
                ),
                strict=False,
            )
        return _VAEEncoder(vae)

    raise ValueError(f"Unsupported encoder: {enc_type}")


# --------------------------------------------------------------------------- public builder
def build_resnet18(
    *,
    # backbone pretraining
    pretrained_backbone: bool = True,
    pretrained: bool | None = None,
    # pooling encoder type
    encoder_type: EncoderType = "ae",
    pool_type: EncoderType | None = None,
    custom_encoder: Optional[nn.Module] = None,
    # freezing options
    freeze_encoder: bool = False,
    freeze_pool: bool | None = None,
    # latent dim
    latent_dim: int = 128,
    latent: int | None = None,
    # classifier
    num_classes: int = 10,
    # pretrained pool encoder
    pretrained_encoder: bool = False,
    pretrained_pooling_block: bool | None = None,
) -> ResNetWithEncoder:
    # map alias arguments
    if pretrained is not None:
        pretrained_backbone = pretrained
    if pool_type is not None:
        encoder_type = pool_type
    if freeze_pool is not None:
        freeze_encoder = freeze_pool
    if latent is not None:
        latent_dim = latent
    if pretrained_pooling_block is not None:
        pretrained_encoder = pretrained_pooling_block

    # build backbone and extract feature-map dims
    backbone = resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    )
    features = nn.Sequential(*list(backbone.children())[:-2])
    # compute feature-map shape (B, C, H, W)
    with torch.no_grad():
        feat = features(torch.zeros(1, 3, 224, 224))
    feat_ch, feat_h, feat_w = feat.shape[1], feat.shape[2], feat.shape[3]

    # build encoder block
    encoder = custom_encoder or _build_encoder(
        encoder_type, feat_ch, latent_dim, pretrained_encoder
    )
    # determine classifier input dim by dummy passing through features + encoder
    with torch.no_grad():
        dummy_feats = features(torch.zeros(1, 3, 224, 224))
        enc_out = encoder(dummy_feats)
        z_latent = enc_out[-1] if isinstance(enc_out, tuple) else enc_out
        clf_dim = torch.flatten(z_latent, 1).shape[1]
    model = ResNetWithEncoder(backbone, encoder, clf_dim, num_classes, freeze_encoder)
    return model
