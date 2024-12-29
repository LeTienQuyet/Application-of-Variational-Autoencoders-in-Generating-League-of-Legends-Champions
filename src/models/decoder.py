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
        self.active_func = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.trans_conv1(x)
        x = self.active_func(x)
        x = self.trans_conv2(x)
        x = self.active_func(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.linear_1 = nn.Linear(
            in_features=latent_dim,
            out_features=4096
        )
        self.linear_2 = nn.Linear(
            in_features=4096,
            out_features=25088
        )
        self.dropout = nn.Dropout(p=0.5)
        self.decoder_block_1 = Decoder_Block(
            in_channels=512, out_channels=256,
            kernel_size=3, stride=2, padding=1,
            output_padding_1=1, output_padding_2=0
        )
        self.decoder_block_2 = Decoder_Block(
            in_channels=256, out_channels=128,
            kernel_size=3, stride=2, padding=1,
            output_padding_1=1, output_padding_2=0
        )
        self.decoder_block_3 = Decoder_Block(
            in_channels=128, out_channels=64,
            kernel_size=3, stride=2, padding=1,
            output_padding_1=1, output_padding_2=0
        )
        self.decoder_block_4= Decoder_Block(
            in_channels=64, out_channels=32,
            kernel_size=3, stride=2, padding=1,
            output_padding_1=1, output_padding_2=0
        )
        self.decoder_block_5 = Decoder_Block(
            in_channels=32, out_channels=3,
            kernel_size=3, stride=2, padding=1,
            output_padding_1=1, output_padding_2=0
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        "Sampling reparameterize shape = (latent_dim = 256)"
        x = self.linear_1(x)             # -> (4096)
        x = self.dropout(x)              # -> (4096)
        x = self.linear_2(x)             # -> (25088)
        x = x.view(x.size(0), 512, 7, 7) # -> (7, 7, 512)
        x = self.decoder_block_1(x)      # -> (14, 14, 256)
        x = self.decoder_block_2(x)      # -> (28, 28, 128)
        x = self.decoder_block_3(x)      # -> (56, 56, 64)
        x = self.decoder_block_4(x)      # -> (112, 112, 32)
        x = self.decoder_block_5(x)      # -> (224, 224, 3)
        x = self.sigmoid(x)
        return x