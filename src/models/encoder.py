from torchvision.models import resnet50, ResNet50_Weights

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=1024, input_dim=4):
        super(Encoder, self).__init__()

        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        original_conv1 = self.base_model.conv1
        self.base_model.conv1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )

        with torch.no_grad():
            self.base_model.conv1.weight[:, :3, :, :] = original_conv1.weight
            nn.init.xavier_uniform_(self.base_model.conv1.weight[:, 3:, :, :])

        self.base_model.fc = nn.Identity()
        self.fc = nn.Linear(
            in_features=2048,
            out_features=4096
        )
        self.dropout = nn.Dropout(p=0.5)

        self.mu = nn.Linear(
            in_features=4096,
            out_features=latent_dim
        )
        self.log_var = nn.Linear(
            in_features=4096,
            out_features=latent_dim
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.dropout(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        log_var = F.softplus(log_var)
        return mu, log_var