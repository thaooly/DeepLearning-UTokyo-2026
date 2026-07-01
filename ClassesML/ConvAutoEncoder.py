import torch.nn as nn
from ClassesML.Blocks import Conv2DBlock, ConvTranspose2DBlock


class ConvAutoEncoder(nn.Module):
    """
    Convolutional AutoEncoder for spectrograms.

    We use a CNN instead of a flatten+MLP AE

    Encoder: 3 stride-2 convs, halves F and T each time -> latent (c*4, F/8, T/8)
    Decoder: 3 transpose convs, mirrors the encoder back to (1, F, T)

    ae_base_channels (default 16) sets the first stage width c,
    later stages use c*2 and c*4.
    """

    def __init__(self, hyperparameters=None):
        nn.Module.__init__(self)

        if hyperparameters is None:
            hyperparameters = {}

        c          = hyperparameters.get("ae_base_channels", 16)
        activation = nn.LeakyReLU(0.2)

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        self.encoder_layers = nn.ModuleList()

        layer = Conv2DBlock(in_channels=1, out_channels=c,
                            kernel_size=(3, 3), activation=activation,
                            batch_normalization=True, stride=2)
        self.encoder_layers.append(layer)

        layer = Conv2DBlock(in_channels=c, out_channels=c * 2,
                            kernel_size=(3, 3), activation=activation,
                            batch_normalization=True, stride=2)
        self.encoder_layers.append(layer)

        layer = Conv2DBlock(in_channels=c * 2, out_channels=c * 4,
                            kernel_size=(3, 3), activation=activation,
                            batch_normalization=True, stride=2)
        self.encoder_layers.append(layer)

        self.encoder = nn.Sequential(*self.encoder_layers)

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        self.decoder_layers = nn.ModuleList()

        layer = ConvTranspose2DBlock(in_channels=c * 4, out_channels=c * 2,
                                     kernel_size=(3, 3), activation=activation,
                                     batch_normalization=True)
        self.decoder_layers.append(layer)

        layer = ConvTranspose2DBlock(in_channels=c * 2, out_channels=c,
                                     kernel_size=(3, 3), activation=activation,
                                     batch_normalization=True)
        self.decoder_layers.append(layer)

        layer = ConvTranspose2DBlock(in_channels=c, out_channels=1,
                                     kernel_size=(3, 3), activation=activation,
                                     batch_normalization=False)
        self.decoder_layers.append(layer)

        self.decoder_layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*self.decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # crop is needed because 3x stride-2 on an odd dimension (e.g. T=301)
        # doesn't come back to the exact same size after the transpose convs
        target_f = x.shape[2]
        target_t = x.shape[3]
        z     = self.encode(x)
        x_hat = self.decode(z)
        x_hat = x_hat[:, :, :target_f, :target_t]
        return x_hat
