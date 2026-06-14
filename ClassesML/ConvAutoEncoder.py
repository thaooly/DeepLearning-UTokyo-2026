import torch.nn as nn
from ClassesML.Blocks import Conv2DBlock, ConvTranspose2DBlock


class ConvAutoEncoder(nn.Module):
    """
    Convolutional AutoEncoder for 2-D audio spectrograms.

    Architecture:

        Encoder — three Conv2DBlock stages, each halving F and T via stride=2:
            (1,   F,   T  ) -> Conv2DBlock(1  ->c,   stride=2) -> (c,   F/2, T/2)
            (c,   F/2, T/2) -> Conv2DBlock(c  ->c*2, stride=2) -> (c*2, F/4, T/4)
            (c*2, F/4, T/4) -> Conv2DBlock(c*2->c*4, stride=2) -> (c*4, F/8, T/8)  <- latent

        Decoder — three ConvTranspose2DBlock stages, each doubling F and T:
            (c*4, F/8, T/8) -> ConvTranspose2DBlock(c*4->c*2) -> (c*2, F/4, T/4)
            (c*2, F/4, T/4) -> ConvTranspose2DBlock(c*2->c  ) -> (c,   F/2, T/2)
            (c,   F/2, T/2) -> ConvTranspose2DBlock(c  ->1  ) -> (1,   F,   T  )

        A final Tanh activation normalises the output to [-1, 1], matching
        the z-score normalised input spectrograms produced by the dataloaders.

    Args:
        hyperparameters : dict with optional key:
                            "ae_base_channels" — number of channels in the first
                            encoder stage (default: 16). Subsequent stages use
                            2x and 4x this value. Controls the model capacity.
    """

    def __init__(self, hyperparameters=None):
        nn.Module.__init__(self)

        if hyperparameters is None:
            hyperparameters = {}

        c          = hyperparameters.get("ae_base_channels", 16)
        activation = nn.LeakyReLU(0.2)

        # ------------------------------------------------------------------
        # Encoder
        # Conv2DBlock with stride=2 halves F and T at each stage.
        # LeakyReLU avoids dying-ReLU in the encoder where negative
        # activations carry useful gradient signal.
        # ------------------------------------------------------------------
        self.encoder_layers = nn.ModuleList()

        # Stage 1 : (B, 1,   F,   T  ) -> (B, c,   F/2, T/2)
        layer = Conv2DBlock(in_channels=1, out_channels=c,
                            kernel_size=(3, 3), activation=activation,
                            batch_normalization=True, stride=2)
        self.encoder_layers.append(layer)

        # Stage 2 : (B, c,   F/2, T/2) -> (B, c*2, F/4, T/4)
        layer = Conv2DBlock(in_channels=c, out_channels=c * 2,
                            kernel_size=(3, 3), activation=activation,
                            batch_normalization=True, stride=2)
        self.encoder_layers.append(layer)

        # Stage 3 : (B, c*2, F/4, T/4) -> (B, c*4, F/8, T/8)  <- latent space
        layer = Conv2DBlock(in_channels=c * 2, out_channels=c * 4,
                            kernel_size=(3, 3), activation=activation,
                            batch_normalization=True, stride=2)
        self.encoder_layers.append(layer)

        self.encoder = nn.Sequential(*self.encoder_layers)

        # ------------------------------------------------------------------
        # Decoder
        # ConvTranspose2DBlock doubles F and T at each stage, mirroring
        # the encoder symmetrically.
        # ------------------------------------------------------------------
        self.decoder_layers = nn.ModuleList()

        # Stage 1 : (B, c*4, F/8, T/8) -> (B, c*2, F/4, T/4)
        layer = ConvTranspose2DBlock(in_channels=c * 4, out_channels=c * 2,
                                     kernel_size=(3, 3), activation=activation,
                                     batch_normalization=True)
        self.decoder_layers.append(layer)

        # Stage 2 : (B, c*2, F/4, T/4) -> (B, c, F/2, T/2)
        layer = ConvTranspose2DBlock(in_channels=c * 2, out_channels=c,
                                     kernel_size=(3, 3), activation=activation,
                                     batch_normalization=True)
        self.decoder_layers.append(layer)

        # Stage 3 : (B, c, F/2, T/2) -> (B, 1, F, T)
        layer = ConvTranspose2DBlock(in_channels=c, out_channels=1,
                                     kernel_size=(3, 3), activation=activation,
                                     batch_normalization=False)
        self.decoder_layers.append(layer)

        self.decoder_layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*self.decoder_layers)

    def encode(self, x):
        """
        Compress a batch of spectrograms to their latent representation.

        Args:
            x : Tensor (B, 1, F, T)

        Returns:
            Tensor (B, c*4, F/8, T/8)
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Reconstruct spectrograms from a latent representation.

        Args:
            z : Tensor (B, c*4, F/8, T/8)

        Returns:
            Tensor (B, 1, F, T)
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Full encode -> decode pass. Used during pre-training (MSE reconstruction
        loss) and at inference time to produce denoised spectrograms.

        Args:
            x : Tensor (B, 1, F, T)

        Returns:
            Tensor (B, 1, F, T) — reconstructed spectrogram
        """
        target_f = x.shape[2]
        target_t = x.shape[3]
        z     = self.encode(x)
        x_hat = self.decode(z)
        x_hat = x_hat[:, :, :target_f, :target_t]
        return x_hat
