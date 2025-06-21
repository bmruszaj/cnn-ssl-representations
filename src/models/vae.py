import torch
import torch.nn as nn
from typing import Tuple

class ConvVEncoder(nn.Module):

    def __init__(self, in_ch, hidden_ch, latent_dim):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, hidden_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_ch), nn.ReLU(True),
        )

        self.flatten = nn.Flatten()
        flat_dim = hidden_ch * 14 * 14
        self.to_mu     = nn.Linear(flat_dim, latent_dim)
        self.to_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        x = self.conv_blocks(x)
        flat = self.flatten(x)
        mu = self.to_mu(flat)
        logvar = self.to_logvar(flat)

        return mu, logvar

class ConvVDecoder(nn.Module):

    def __init__(self, latent_dim, hidden_ch, out_ch):
        super().__init__()

        self.hidden_ch = hidden_ch

        self.fc = nn.Linear(latent_dim, hidden_ch * 14 * 14)
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(hidden_ch, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, out_ch, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), self.hidden_ch, 14, 14)
        return self.deconv_blocks(x)

class VEncoder(nn.Module):

    def __init__(
            self,
            n_input_features: int,
            n_hidden_neurons: int,
            n_latent_features: int,
    ):
        super().__init__()

        self.input_to_hidden = nn.Linear(n_input_features, n_hidden_neurons)

        self.mu_layer = nn.Linear(n_hidden_neurons, n_latent_features)
        self.log_var_layer = nn.Linear(n_hidden_neurons, n_latent_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = nn.functional.relu(self.input_to_hidden(x))

        return self.mu_layer(hidden), self.log_var_layer(hidden)


class VDecoder(nn.Module):

    def __init__(
            self,
            n_latent_features: int,
            n_hidden_neurons: int,
            n_output_features: int,
    ):
        super().__init__()

        self.latent_to_hidden = nn.Linear(n_latent_features, n_hidden_neurons)
        self.hidden_to_output = nn.Linear(n_hidden_neurons, n_output_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = nn.functional.relu(self.latent_to_hidden(z))

        return torch.sigmoid(self.hidden_to_output(hidden))



class VAE(nn.Module):

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        latent_dim: int,
        beta: float = 1.0
    ):
        super().__init__()

        self.encoder = ConvVEncoder(in_ch, hidden_ch, latent_dim)
        self.decoder = ConvVDecoder(latent_dim, hidden_ch, in_ch)

        self.beta = beta
        self.mu = None
        self.logvar = None

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x: torch.Tensor):

        mu, logvar = self.encoder(x)
        self.mu, self.logvar = mu, logvar

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

    def forward(self,
                model: VAE,
                recon_x: torch.Tensor,
                x: torch.Tensor
                ) -> torch.Tensor:

        recon_loss = self.recon_crit(recon_x, x)
        kld_loss = -0.5 * torch.mean(1 + model.logvar - model.mu.pow(2) - torch.exp(model.logvar))

        return recon_loss + self.beta * kld_loss
