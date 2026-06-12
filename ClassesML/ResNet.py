import torch
import torch.nn as nn
from torch._library import infer_schema

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

        self.layers = nn.ModuleList()

        layers = Conv2DBlock(in_channels=self.input_dim[0],
                             out_channels=self.filters[0],
                             kernel_size=self.kernel_size,
                             activation=Utilities.get_activation(self.activation))
        self.layers.append(layers)
        # layer = nn.MaxPool2d(kernel_size=(2,2))
        # self.layers.append(layer)
        for i in range(4):
            in_channel = self.filters[i]
            out_channel = self.filters[i+1]
            layer = BasicResNetBlock(in_channels=in_channel,
                                      out_channels=out_channel,
                                      kernel_size=self.kernel_size,
                                      activation=Utilities.get_activation(self.activation))
            self.layers.append(layer)
            layer = BasicResNetBlock(in_channels=out_channel, # adjust dimensions!
                                     out_channels=out_channel,
                                     kernel_size=self.kernel_size,
                                     activation=Utilities.get_activation(self.activation))
            self.layers.append(layer)
        layer = nn.AvgPool2d(kernel_size=(2,2))
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

