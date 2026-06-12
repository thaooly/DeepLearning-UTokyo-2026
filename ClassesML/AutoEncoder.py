import torch
import torch.nn as nn
from ClassesML.Blocks import Conv2DBlock, FlattenDenseBlock, UnFlattenDenseBlock, DenseBlock
from Utilities.Utilities import Utilities


class AutoEncoder(nn.Module):
    def __init__(self, hyperparameter):

        nn.Module.__init__(self)
        self.hyperparameters = hyperparameter
        self.hidden_layers_size = hyperparameter["hidden_layers_sizes"]
        self.activation = hyperparameter["activation"]

        self.batch_normalization = hyperparameter["batch_normalization"]
        self.dropout_rate = hyperparameter["dropout_rate"]

        self.input_dim = hyperparameter["input_dim"]
        self.output_dim = hyperparameter["output_dim"]

        self.latent_dim = hyperparameter["latent_dim"]
        self.output_activation = hyperparameter["output_activation"]

        self.n_dense_layer = len(self.hidden_layers_size)

        #-------------------------------------------
        # create Encoder

        self.encoder_layers = nn.ModuleList()

        layer = FlattenDenseBlock(in_size=self.input_dim, out_size=self.hidden_layers_size[0],
                                    activation=Utilities.get_activation(self.activation),
                                    batch_normalization=self.batch_normalization,
                                    dropout_rate=self.dropout_rate)
        self.encoder_layers.append(layer)

        for i in range(self.n_dense_layer - 1):
            layer = DenseBlock(in_size=self.hidden_layers_size[i], out_size=self.hidden_layers_size[i + 1],
                               activation=Utilities.get_activation(self.activation),
                               batch_normalization=self.batch_normalization,
                               dropout_rate=self.dropout_rate)
            self.encoder_layers.append(layer)

        # -------------------------------------------
        # latent space

        layer = nn.Linear(self.hidden_layers_size[-1], self.latent_dim[0])
        self.encoder_layers.append(layer)

        # -------------------------------------------
        # create decoder

        self.decoder_layers = nn.ModuleList()

        units = self.latent_dim + list(self.hidden_layers_size)[::-1]

        for i in range(0, len(units) - 1):
            layer = DenseBlock(in_size=units[i], out_size=units[i + 1],
                               activation=Utilities.get_activation(self.activation),
                               batch_normalization=self.batch_normalization,
                               dropout_rate=self.dropout_rate)
            self.decoder_layers.append(layer)

        layer = UnFlattenDenseBlock(in_size=units[-1], out_size=self.input_dim,
                           activation=Utilities.get_activation(self.output_activation),
                           batch_normalization=self.batch_normalization,
                           dropout_rate=self.dropout_rate)
        self.decoder_layers.append(layer)

        # put it all together in the exact way we have been setting it up in the previous code
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

        # ------------------------------------------

    def encode(self, x):
        z = self.encoder(x)
        return z

    def sample(self, z):
        with torch.no_grad():
            x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat