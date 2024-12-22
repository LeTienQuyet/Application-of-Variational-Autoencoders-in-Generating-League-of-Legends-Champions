import torch
import torch.nn as nn

class Decoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(Decoder_Block, self).__init__()
        self.trans_conv1 = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=int(in_channels/2),
            kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=output_padding,
            bias=False
        )
        self.trans_conv2 = nn.ConvTranspose2d(
            in_channels=int(in_channels/2), out_channels=out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=output_padding,
            bias=False
        )
        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels,
            affine=True
        )
        self.active_func = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.trans_conv1(x)
        x = self.trans_conv2(x)
        x = self.batch_norm(x)
        x = self.active_func(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.linear_1 = nn.Linear(
            in_features=latent_dim,
            out_features=128
        )
        self.linear_2 = nn.Linear(
            in_features=128,
            out_features=256
        )
        self.linear_3 = nn.Linear(
            in_features=256,
            out_features=3200
        )
        self.dropout_1 = nn.Dropout(p=0.25)
        self.dropout_2 = nn.Dropout(p=0.25)
        self.dropout_3 = nn.Dropout(p=0.25)
        self.unpooling = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )
        self.decoder_block_1 = Decoder_Block(
            in_channels=128, out_channels=32,
            kernel_size=3, stride=2,
            padding=1, output_padding=1
        )
        self.decoder_block_2 = Decoder_Block(
            in_channels=32, out_channels=8,
            kernel_size=3, stride=2,
            padding=1, output_padding=1
        )
        self.decoder_block_3 = Decoder_Block(
            in_channels=8, out_channels=3,
            kernel_size=3, stride=2,
            padding=1, output_padding=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        "Sampling reparameterize shape = (latent_dim = 64)"
        x = self.linear_1(x)             # -> (128)
        x = self.dropout_1(x)            # -> (128)
        x = self.linear_2(x)             # -> (256)
        x = self.dropout_2(x)            # -> (256)
        x = self.linear_3(x)             # -> (3200)
        x = self.dropout_3(x)            # -> (3200)
        x = x.view(x.size(0), 128, 5, 5) # -> (5, 5, 128)
        x = self.unpooling(x)            # -> (10, 10, 128)
        x = self.decoder_block_1(x)      # -> (40, 40, 32)
        x = self.decoder_block_2(x)      # -> (160, 160, 8)
        x = self.decoder_block_3(x)      # -> (640, 640, 3)
        x = self.sigmoid(x)              # -> (640, 640, 3)
        return x