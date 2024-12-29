from torchvision.models import vgg16

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.backbone = vgg16(weights="IMAGENET1K_V1")
        self.mu = nn.Linear(
            in_features=1000,
            out_features=latent_dim
        )
        self.log_var = nn.Linear(
            in_features=1000,
            out_features=latent_dim
        )

    def forward(self, x):
        x = self.backbone(x)
        mu = self.mu(x)             # -> (latent_dim=256)
        log_var = self.log_var(x)   # -> (latent_dim=256)
        return mu, log_var