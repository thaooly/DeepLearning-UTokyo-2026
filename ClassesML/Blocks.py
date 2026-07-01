import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import Flatten

class DenseBlock(nn.Module):

    def __init__(self, in_size, out_size,
                 activation=nn.ReLU(),
                 batch_normalization=False,
                 dropout_rate=0.1,):

        super(DenseBlock, self).__init__()

        self.linear_layer = nn.Linear(in_size, out_size)
        self.activation = activation

        if batch_normalization:
            self.batch_norm_layer = nn.BatchNorm1d(out_size)
        else:
            self.batch_norm_layer = None

        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):

        x = self.linear_layer(x)
        if self.batch_norm_layer is not None:
            x = self.batch_norm_layer(x)
        x = self.activation(x)
        x = self.dropout_layer(x)

        return x


class FlattenDenseBlock(nn.Module):

    def __init__(self, in_size, out_size, activation=nn.ReLU(),
                 batch_normalization=False,
                 dropout_rate=0.1,):
        super(FlattenDenseBlock, self).__init__()

        in_size_flatten = np.prod(in_size)
        self.flatten_layer = nn.Flatten()
        self.dense_layer = DenseBlock(in_size=in_size_flatten,
                                      out_size=out_size,
                                      activation=activation,
                                      batch_normalization=batch_normalization,
                                      dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        return x


class Conv2DBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size,
                 activation=nn.ReLU(),
                 batch_normalization=False,
                 dropout_rate=0.1, stride=1):

        super(Conv2DBlock, self).__init__()
        # stride=1 keeps padding='same' (output size = input size)
        # stride>1 needs explicit padding instead, used for downsampling in the AE encoder
        self.conv_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
            padding=1 if stride > 1 else 'same')
        self.activation = activation
        self.batch_norm_layer = nn.BatchNorm2d(out_channels) if batch_normalization else None
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv_layer(x)
        if self.batch_norm_layer:
            x = self.batch_norm_layer(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        return x

class BasicResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size,
                 activation=nn.ReLU(),
                 batch_normalization=False,
                 dropout_rate=0.1):

        super(BasicResNetBlock, self).__init__()

        self.conv_layer_1 = Conv2DBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        activation=activation,
                                        batch_normalization=batch_normalization,
                                        dropout_rate=dropout_rate)

        self.conv_layer_2 = Conv2DBlock(in_channels=out_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        activation=activation,
                                        batch_normalization=batch_normalization,
                                        dropout_rate=dropout_rate)

        # identity shortcut when channels match, 1x1 conv otherwise to match dimensions
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            shortcut_layers = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=not batch_normalization,
                )
            ]

            if batch_normalization:
                shortcut_layers.append(nn.BatchNorm2d(out_channels))

            self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        residual = x
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        residual = self.shortcut(residual)
        x = x + residual
        return x

class UnFlattenDenseBlock(nn.Module):
    def __init__(self, in_size, out_size,activation=nn.ReLU(),
                 batch_normalization=False, dropout_rate=0.1):
        super(UnFlattenDenseBlock, self).__init__()
        self.dense_layer = DenseBlock(in_size=in_size,
                                      out_size=np.prod(out_size),
                                      activation=activation,
                                      batch_normalization=batch_normalization,
                                      dropout_rate=dropout_rate)

        self.unflatten_layer = nn.Unflatten(dim=1, unflattened_size=out_size)

    def forward(self, x):
        x = self.dense_layer(x)
        x = self.unflatten_layer(x)
        return x

class ConvTranspose2DBlock(nn.Module):
    # Conv2DBlock for the AE decoder, always upsamples by 2
    def __init__(self, in_channels, out_channels, kernel_size, activation=nn.ReLU,
                 batch_normalization=False, dropout_rate=0.1):
        super(ConvTranspose2DBlock, self).__init__()
        self.activation = activation
        self.conv_layer = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=2, padding=1, output_padding=1
        )
        self.batch_norm_layer = nn.BatchNorm2d(out_channels) if batch_normalization else None
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv_layer(x)
        if self.batch_norm_layer:
            x = self.batch_norm_layer(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al., 2018).
    Learns a per-channel weight so the network can amplify informative
    frequency channels and suppress noisy ones, instead of treating all
    channels equally.

    squeeze: global avg pool (B,C,F,T) -> (B,C,1,1)
    excitation: 2 FC layers with bottleneck -> per-channel weight in (0,1)
    scale: multiply input by its channel weights
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        bottleneck = max(channels // reduction, 1)
        self.squeeze    = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.squeeze(x)                        # (B, C, 1, 1)
        w = self.excitation(w).view(b, c, 1, 1)   # (B, C, 1, 1)
        return x * w


class TFAttentionBlock(nn.Module):
    """
    Time-Frequency attention block (inspired by CBAM, Woo et al., 2018).
    Two separate 1D attention masks, one over frequency bins and one over
    time frames, so the network can ignore silent frames and out-of-band
    frequencies regardless of which channel they're in.
    """

    def __init__(self):
        super().__init__()
        self.freq_conv = nn.Conv2d(1, 1, kernel_size=(7, 1), padding=(3, 0), bias=False)
        self.time_conv = nn.Conv2d(1, 1, kernel_size=(1, 7), padding=(0, 3), bias=False)

    def forward(self, x):
        # average over channels to get a single (F,T) map
        x_avg = x.mean(dim=1, keepdim=True)              # (B, 1, F, T)

        # frequency mask: avg over time, conv over F
        freq_w = torch.sigmoid(
            self.freq_conv(x_avg.mean(dim=3, keepdim=True))
        )                                                 # (B, 1, F, 1)

        # time mask: avg over freq, conv over T
        time_w = torch.sigmoid(
            self.time_conv(x_avg.mean(dim=2, keepdim=True))
        )                                                 # (B, 1, 1, T)

        return x * freq_w * time_w
