import torch
import torch.nn as nn

class Encoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Encoder_Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=int(out_channels/2),
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=True
        )
        self.conv2 = nn.Conv2d(
            in_channels=int(out_channels/2), out_channels=out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=True
        )
        self.bn1 = nn.BatchNorm2d(num_features=int(out_channels/2))
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.active_func = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.active_func(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.active_func(x)
        x = self.pooling(x)
        return x

class Encoder(nn.Module):
    def __init__(self, latent_dim=1024, input_dim=4):
        super(Encoder, self).__init__()

        num_channels = [input_dim, 8, 32, 128, 512]

        self.feature = nn.ModuleList([
            Encoder_Block(
                in_channels=num_channels[i], out_channels=num_channels[i+1],
                kernel_size=3, stride=1, padding=1
            ) for i in range(len(num_channels)-1)
        ])
        self.pooling = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.linear = nn.Linear(
            in_features=25088,
            out_features=4096,
        )
        self.active_func = nn.ReLU(inplace=True)
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
        "Image shape = (224, 224, [4, 3])"
        for encoder_block in self.feature:
            x = encoder_block(x)    # -> (112, 112, 8) -> (56, 56, 32) -> (28, 28, 128) -> (14, 14, 512)
        x = self.pooling(x)         # -> (7, 7, 512)
        x = x.view(x.size(0), -1)   # -> (25088)
        x = self.linear(x)          # -> (4096)
        x = self.active_func(x)     # -> (4096)
        x = self.dropout(x)         # -> (4096)
        mu = self.mu(x)             # -> (latent_dim = 1024)
        log_var = self.log_var(x)   # -> (latent_dim = 1024)
        return mu, log_var