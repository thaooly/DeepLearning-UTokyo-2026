import torch
import torch.nn as nn

from ClassesML.Blocks import DenseBlock, Conv2DBlock, BasicResNetBlock
from Utilities.Utilities import Utilities

class ResNet(nn.Module):

    def __init__(self, hyperparameters):

        nn.Module.__init__(self)

        self.hidden_layers_size = hyperparameters["hidden_layers_size"]
        self.activation = hyperparameters["activation"]
        self.batch_normalization = hyperparameters["batch_normalization"]
        self.dropout_rate = hyperparameters["dropout_rate"]
        self.input_dim = hyperparameters["input_dim"]
        self.output_dim = hyperparameters["output_dim"]
        self.kernel_size = hyperparameters["kernel_size"]
        self.filters = hyperparameters["filters"]
        self.n_dense_layer = len(self.hidden_layers_size)
        self.n_conv_layer = len(self.filters)
        self.pool_every_stage = hyperparameters.get("pool_every_stage", False)
        self.pool_output_size = hyperparameters.get("pool_output_size", None)

        self.layers = nn.ModuleList()

        layers = Conv2DBlock(in_channels=self.input_dim[0],
                             out_channels=self.filters[0],
                             kernel_size=self.kernel_size,
                             activation=Utilities.get_activation(self.activation),
                             batch_normalization=self.batch_normalization,
                             dropout_rate=self.dropout_rate)
        self.layers.append(layers)

        for i in range(4):
            in_channel = self.filters[i]
            out_channel = self.filters[i+1]
            layer = BasicResNetBlock(in_channels=in_channel,
                                      out_channels=out_channel,
                                      kernel_size=self.kernel_size,
                                      activation=Utilities.get_activation(self.activation),
                                      batch_normalization=self.batch_normalization,
                                      dropout_rate=self.dropout_rate)
            self.layers.append(layer)
            layer = BasicResNetBlock(in_channels=out_channel, # adjust dimensions!
                                     out_channels=out_channel,
                                     kernel_size=self.kernel_size,
                                     activation=Utilities.get_activation(self.activation),
                                     batch_normalization=self.batch_normalization,
                                     dropout_rate=self.dropout_rate)
            self.layers.append(layer)

            if self.pool_every_stage:
                self.layers.append(nn.MaxPool2d(kernel_size=(2, 2)))

        if self.pool_output_size is None:
            layer = nn.AvgPool2d(kernel_size=(2,2))
        else:
            layer = nn.AdaptiveAvgPool2d(self.pool_output_size)
        self.layers.append(layer)

        layer = nn.Flatten()
        self.layers.append(layer)
        layer = nn.LazyLinear(out_features=self.hidden_layers_size[0])
        self.layers.append(layer)
        layer = DenseBlock(in_size=self.hidden_layers_size[0], out_size=self.hidden_layers_size[1],
                           activation=Utilities.get_activation(self.activation),
                           batch_normalization=self.batch_normalization,
                           dropout_rate=self.dropout_rate)
        self.layers.append(layer)
        layer = nn.Linear(in_features=self.hidden_layers_size[-1], out_features=self.output_dim)
        self.layers.append(layer)
        self.classifier = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.classifier(x)
        return x
