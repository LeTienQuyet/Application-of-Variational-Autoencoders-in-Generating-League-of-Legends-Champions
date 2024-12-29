from src.models.encoder import Encoder
from src.models.decoder import Decoder

import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim, num_channels):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim, num_channels)
        self.decoder = Decoder(latent_dim, num_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar