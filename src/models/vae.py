import torch
import torch.nn as nn
from typing import Tuple


class ConvVEncoder(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, latent_dim: int):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_ch), nn.ReLU(True),
        )
        self.to_mu = nn.Conv2d(hidden_ch, latent_dim, kernel_size=1)
        self.to_logvar = nn.Conv2d(hidden_ch, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_blocks(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar


class ConvVDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_ch: int, out_ch: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_ch), nn.ReLU(True),
            nn.Conv2d(hidden_ch, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, out_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        latent_dim: int,
        beta: float = 1.0,
    ):
        super().__init__()
        self.encoder = ConvVEncoder(in_ch, hidden_ch, latent_dim)
        self.decoder = ConvVDecoder(latent_dim, hidden_ch, in_ch)

        self.beta = beta
        self.mu: torch.Tensor | None = None
        self.logvar: torch.Tensor | None = None

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encoder(x)
        self.mu = mu
        self.logvar = logvar
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon

    def kl_divergence(self) -> torch.Tensor:
        return -0.5 * torch.mean(1 + self.logvar - self.mu.pow(2) - torch.exp(self.logvar))


class VAELoss(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.recon_crit = nn.MSELoss(reduction='mean')
        self.beta = beta

    def forward(
        self,
        model: VAE,
        recon_x: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        recon_loss = self.recon_crit(recon_x, x)
        kld_loss = model.kl_divergence()
        return recon_loss + self.beta * kld_loss
