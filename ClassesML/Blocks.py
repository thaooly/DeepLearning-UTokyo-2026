import torch
import torch.nn as nn
import numpy as np
import math
#from torch.nn import Flatten

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
        self.conv_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,                              # ← utiliser stride
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

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = Conv2DBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        activation=activation,
                                        batch_normalization=batch_normalization,
                                        dropout_rate=dropout_rate)

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
    Squeeze-and-Excitation Block (Hu et al., 2018).

    Recalibrates channel-wise feature responses by explicitly modelling
    inter-channel dependencies through a lightweight gating mechanism:

        1. Squeeze   : global average pooling collapses spatial dimensions
                       to a channel descriptor vector of shape (B, C).
        2. Excitation: two fully-connected layers with a bottleneck learn
                       a channel-wise gating vector in (0, 1)^C.
        3. Scale     : each channel of the input feature map is multiplied
                       by its corresponding gate value.

    Motivation for bird audio classification:
        Different frequency channels in a spectrogram carry vastly different
        amounts of discriminative information — the fundamental frequency band
        of a species is informative while adjacent noise bands are not. The SE
        block learns to up-weight informative channels and suppress others,
        with negligible parameter overhead (2 × C²/reduction weights per block).

    Args:
        channels  : number of input (and output) channels C.
        reduction : bottleneck compression factor (default: 4).
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
    Time-Frequency Attention Block (inspired by CBAM, Woo et al., 2018).

    Applies two independent attention masks over the spectrogram axes:

        - Frequency attention : collapses the time axis by average pooling,
          applies a vertical convolution (7×1), and produces a per-frequency-bin
          gate of shape (B, 1, F, 1).

        - Time attention      : collapses the frequency axis by average pooling,
          applies a horizontal convolution (1×7), and produces a per-frame
          gate of shape (B, 1, 1, T).

    The two masks are multiplied element-wise onto the input feature map,
    allowing the network to jointly focus on the most relevant frequency bands
    and temporal frames.

    Motivation for bird audio classification:
        Bird vocalisations are localised both in frequency (species-specific
        pitch range) and in time (discrete syllables separated by silence).
        Dual-axis attention directly encodes this physical prior, enabling
        the classifier to suppress uninformative background regions of the
        spectrogram without increasing the receptive field of the CNN.
    """

    def __init__(self):
        super().__init__()
        self.freq_conv = nn.Conv2d(1, 1, kernel_size=(7, 1), padding=(3, 0), bias=False)
        self.time_conv = nn.Conv2d(1, 1, kernel_size=(1, 7), padding=(0, 3), bias=False)

    def forward(self, x):
        # Collapse channels to obtain a single-channel spatial map.
        x_avg = x.mean(dim=1, keepdim=True)              # (B, 1, F, T)

        # Frequency attention: pool over time, convolve vertically.
        freq_w = torch.sigmoid(
            self.freq_conv(x_avg.mean(dim=3, keepdim=True))
        )                                                 # (B, 1, F, 1)

        # Time attention: pool over frequency, convolve horizontally.
        time_w = torch.sigmoid(
            self.time_conv(x_avg.mean(dim=2, keepdim=True))
        )                                                 # (B, 1, 1, T)

        return x * freq_w * time_w