import torch.nn as nn

from ClassesML.Blocks import DenseBlock, FlattenDenseBlock

from Utilities.Utilities import Utilities

class MLP(nn.Module):

    def __init__(self, hyperparameters):
        nn.Module.__init__(self)

        self.hidden_layers_size = hyperparameters['hidden_layers_size']
        self.activation = hyperparameters['activation']

        self.batch_normalization = hyperparameters['batch_normalization']
        self.dropout_rate = hyperparameters['dropout_rate']

        self.input_dim = hyperparameters['input_dim']
        self.output_dim = hyperparameters['output_dim']

        self.n_dense_layers = len(self.hidden_layers_size)

        self.layers = nn.ModuleList()

        layer = FlattenDenseBlock(in_size=self.input_dim,out_size=self.hidden_layers_size[0],
                                  activation=Utilities.get_activation(self.activation),
                                  batch_normalization=self.batch_normalization,
                                  dropout_rate=self.dropout_rate)
        self.layers.append(layer)

        for i in range(self.n_dense_layers - 1):
            layer = DenseBlock(in_size=self.hidden_layers_size[i],
                               out_size=self.hidden_layers_size[i + 1],
                               activation=Utilities.get_activation(self.activation),
                               batch_normalization=self.batch_normalization,
                               dropout_rate=self.dropout_rate)
            self.layers.append(layer)

        # Define output layer
        layer = nn.Linear(in_features=self.hidden_layers_size[-1],
                          out_features=self.output_dim)
        self.layers.append(layer)

        self.classifier = nn.Sequential(*self.layers)

    def forward(self, x):
        x_hat = self.classifier(x)
        return x_hat