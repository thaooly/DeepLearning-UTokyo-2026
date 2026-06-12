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
                 dropout_rate=0.1):

        super(Conv2DBlock, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    padding="same")
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
                 activation:nn.ReLU(),
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
