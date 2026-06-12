import torch
import torch.nn as nn
from ClassesML.Blocks import DenseBlock, FlattenDenseBlock, Conv2DBlock, UnFlattenDenseBlock
from Utilities.Utilities import Utilities

# generate new data similar to real data that the discriminator has been trained on
# first: produce a fake image based on random noise
# third: train generator
# Generator: noise -> Dense layer (1, 128*4*4) -> Reshape (128 channels, 4x4 dim)
# -> Conv + UpSample (increase pixel by 8x8) -> Conv + UpSample (expand, contrary to MaxPooling) -> Conv -> FakeImage
# Conditional generator: merge random noise vector with condition to ensure desired class is generated

# generally important: dimension matching!

class Generator(nn.Module):
    def __init__(self, hyperparameters):
        nn.Module.__init__(self)
        self.hyperparameters = hyperparameters

        self.activation = self.hyperparameters['activation']
        self.input_dim = self.hyperparameters['input_dim']
        self.n_classes = self.hyperparameters['output_dim']
        self.filters = self.hyperparameters['filters']
        self.kernel_size = self.hyperparameters['kernel_size']
        self.embedding_dim = self.hyperparameters['embedding_dim']
        self.latent_dim = self.hyperparameters['latent_dim']

        self.batch_normalization = self.hyperparameters['batch_normalization']
        self.dropout_rate = self.hyperparameters['dropout_rate']

        self.n_conv_layers = len(self.filters)

        # Embedding
        self.embedding = nn.Embedding(self.n_classes, self.embedding_dim)

        self.layers_generator = nn.ModuleList()
        size = self.input_dim[1]
        for i in range(0, self.n_conv_layers - 1):
            size /= 2

        self.size_reshape_noise = (self.filters[0], int(size), int(size))

        layer = UnFlattenDenseBlock(self.latent_dim + self.embedding_dim,
                                    out_size=self.size_reshape_noise,
                                    batch_normalization=self.batch_normalization,
                                    activation=Utilities.get_activation(self.activation),
                                    dropout_rate=0.0)

        self.layers_generator.append(layer)
        for i in range(0, self.n_conv_layers - 1):
            layer = Conv2DBlock(in_channels=self.filters[i],
                                out_channels=self.filters[i + 1],
                                kernel_size=self.kernel_size,
                                activation=Utilities.get_activation(self.activation),
                                dropout_rate=self.dropout_rate)
            self.layers_generator.append(layer)
            layer = nn.Upsample(scale_factor=(2,2))
            self.layers_generator.append(layer)

        layer = Conv2DBlock(in_channels=self.filters[-1],
                            out_channels=1,
                            kernel_size=self.kernel_size,
                            batch_normalization=False,
                            activation=Utilities.get_activation("sigmoid"),
                            dropout_rate=0.0)
        self.layers_generator.append(layer)
        self.generator = nn.Sequential(*self.layers_generator)

    def forward(self, noise, labels):
        embedded_labels = self.embedding(labels)
        # Concatenate noise and embedded labels
        gen_input = torch.cat((noise, embedded_labels), dim=1)
        img = self.generator(gen_input)
        return img

class Discriminator(nn.Module):
    def __init__(self, hyperparameters):
        nn.Module.__init__(self)
        self.hyperparameters = hyperparameters

        self.activation = self.hyperparameters['discriminator_activation']
        self.input_dim = self.hyperparameters['input_dim']
        self.n_classes = self.hyperparameters['output_dim']
        self.discriminator_filters = self.hyperparameters['discriminator_filters']
        self.kernel_size = self.hyperparameters['kernel_size']
        self.batch_normalization = self.hyperparameters['batch_normalization']
        self.dropout_rate = self.hyperparameters['dropout_rate']
        self.embedding_dim = self.hyperparameters['embedding_dim']
        self.latent_dim = self.hyperparameters['latent_dim']
        self.n_conv_discriminator_layers = len(self.discriminator_filters)
        # Embedding
        self.layers_embedding = nn.ModuleList()
        layer = nn.Embedding(self.n_classes, self.embedding_dim)
        self.layers_embedding.append(layer)

        layer = UnFlattenDenseBlock(self.embedding_dim,
                                    out_size=(self.input_dim[0], self.input_dim[1], self.input_dim[2]),
                                    activation=Utilities.get_activation(self.activation),
                                    batch_normalization=self.batch_normalization,
                                    dropout_rate=self.dropout_rate)
        self.layers_embedding.append(layer)
        self.embedding = nn.Sequential(*self.layers_embedding)

        self.layers_discriminator = nn.ModuleList()

        # Discriminator
        layer = Conv2DBlock(in_channels=self.input_dim[0] + 1,
                            out_channels=self.discriminator_filters[0],
                            kernel_size=self.kernel_size,
                            activation=Utilities.get_activation(self.activation),
                            batch_normalization=self.batch_normalization,
                            dropout_rate=self.dropout_rate)
        self.layers_discriminator.append(layer)
        layer = nn.MaxPool2d(kernel_size=(2,2))
        self.layers_discriminator.append(layer)
        for i in range(0, self.n_conv_discriminator_layers - 1):
            layer = Conv2DBlock(in_channels=self.discriminator_filters[i],
                                out_channels=self.discriminator_filters[i + 1],
                                kernel_size=self.kernel_size,
                                activation=Utilities.get_activation(self.activation),
                                batch_normalization=self.batch_normalization,
                                dropout_rate=self.dropout_rate)

            self.layers_discriminator.append(layer)
            layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.layers_discriminator.append(layer)

        layer = nn.Flatten()
        self.layers_discriminator.append(layer)
        layer = nn.LazyLinear(out_features=128)
        self.layers_discriminator.append(layer)
        layer = DenseBlock(128, 1,
                           activation=Utilities.get_activation('sigmoid'),
                           batch_normalization=False,
                           dropout_rate=0.0)
        self.layers_discriminator.append(layer)
        self.discriminator = nn.Sequential(*self.layers_discriminator)

    def forward(self, images, labels):
        embedded_labels = self.embedding(labels)
        disc_input = torch.concat([images, embedded_labels], dim=1)
        x_hat = self.discriminator(disc_input)
        return x_hat