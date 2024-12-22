import torch
import torch.nn as nn

class Decoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Decoder_Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=int(out_channels/2),
            kernel_size=kernel_size, stride=int(stride/2),
            padding=padding, bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=int(out_channels/2), out_channels=out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels,
            affine=True
        )
        self.active_func = nn.ReLU(inplace=False)
        self.pooling = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm(x)
        x = self.active_func(x)
        x = self.pooling(x)
        return x

class Decoder(nn.Module):
    def __init__(self, laten_dim):
        super(Decoder, self).__init__()
        self.decoder_block_1 = Decoder_Block(
            in_channels=3, out_channels=8,
            kernel_size=3, stride=2, padding=1
        )
        self.decoder_block_2 = Decoder_Block(
            in_channels=8, out_channels=32,
            kernel_size=3, stride=2, padding=1
        )
        self.decoder_block_3 = Decoder_Block(
            in_channels=32, out_channels=128,
            kernel_size=3, stride=2, padding=1
        )
        self.pooling = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.linear_1 = nn.Linear(
            in_features=3200,
            out_features=256,
        )
        self.linear_2 = nn.Linear(
            in_features=256,
            out_features=128,
        )
        self.mu = nn.Linear(
            in_features=128,
            out_features=laten_dim
        )
        self.log_var = nn.Linear(
            in_features=128,
            out_features=laten_dim
        )
        self.dropout_1 = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.dropout_3 = nn.Dropout(p=0.5)

    def forward(self, x):
        "Image shape = (640, 640, 3)"
        x = self.decoder_block_1(x) # -> (160, 160, 8)
        x = self.decoder_block_2(x) # -> (40, 40, 32)
        x = self.decoder_block_3(x) # -> (10, 10, 128)
        x = self.pooling(x)         # -> (5, 5, 128)
        x = x.view(x.size(0), -1)   # -> (3200)
        x = self.dropout_1(x)       # -> (3200)
        x = self.linear_1(x)        # -> (256)
        x = self.dropout_2(x)       # -> (256)
        x = self.linear_2(x)        # -> (128)
        x = self.dropout_3(x)       # -> (128)
        mu = self.mu(x)             # -> (latent_dim=64)
        log_var = self.log_var(x)   # -> (latent_dim=64)
        return mu, log_var