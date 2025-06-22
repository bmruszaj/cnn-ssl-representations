import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_channels: int, out_channels: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.decoder(z)


class Autoencoder(nn.Module):
    """AE pooling block that returns (recon, z) when training, but only recon when eval."""

    def __init__(self, in_channels: int = 512, latent_channels: int = 128) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels)
        self.decoder = Decoder(latent_channels, in_channels)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        if self.training:
            return recon, z
        return recon

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)


class AELoss(nn.Module):
    def __init__(self, recon_loss_weight: float = 1.0):
        super().__init__()
        self.recon_crit = nn.MSELoss(reduction="mean")
        self.recon_loss_weight = recon_loss_weight

    def forward(self, recon_x: Tensor, x: Tensor) -> Tensor:
        recon_loss = self.recon_crit(recon_x, x)
        return self.recon_loss_weight * recon_loss
