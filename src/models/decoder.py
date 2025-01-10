import torch
import torch.nn as nn

class Decoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding_1, output_padding_2):
        super(Decoder_Block, self).__init__()
        self.trans_conv1 = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=output_padding_1,
            bias=True
        )
        self.trans_conv2 = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=int(stride/2),
            padding=padding, output_padding=output_padding_2,
            bias=True
        )
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.active_func = nn.ReLU(inplace=True)
        self.res_connect = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, stride=2,
            padding=0, output_padding=1,
            bias=True
        )

    def forward(self, x):
        x_residual = self.res_connect(x)
        x = self.trans_conv1(x)
        x = self.bn1(x)
        x = self.active_func(x)
        x = self.trans_conv2(x)
        x = self.bn2(x)
        x = self.active_func(x)
        x = x + x_residual
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=1024, input_dim=4):
        super(Decoder, self).__init__()

        num_channels = [512, 256, 128, 64, 32, input_dim]

        self.feature = nn.ModuleList([
            Decoder_Block(
                in_channels=num_channels[i], out_channels=num_channels[i+1],
                kernel_size=3, stride=2, padding=1,
                output_padding_1=1, output_padding_2=0
            ) for i in range(len(num_channels)-1)
        ])
        self.linear_1 = nn.Linear(
            in_features=latent_dim,
            out_features=4096
        )
        self.linear_2 = nn.Linear(
            in_features=4096,
            out_features=25088
        )
        self.active_func = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        "Sampling reparameterize shape = (latent_dim = 1024)"
        x = self.linear_1(x)             # -> (4096)
        x = self.active_func(x)          # -> (4096)
        x = self.dropout(x)              # -> (4096)
        x = self.linear_2(x)             # -> (25088)
        x = self.active_func(x)          # -> (25088)
        x = self.dropout(x)              # -> (25088)
        x = x.view(x.size(0), 512, 7, 7) # -> (7, 7, 512)
        for decoder_block in self.feature:
            x = decoder_block(x)         # -> (14, 14, 256) -> (28, 28, 128) -> (56, 56, 64) -> (112, 112, 32) -> (224, 224, [4, 3])
        x = self.sigmoid(x)
        return x