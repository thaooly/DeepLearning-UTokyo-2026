import torch.nn as nn
from ClassesML.Blocks import DenseBlock, Conv2DBlock, BasicResNetBlock, SEBlock, TFAttentionBlock
from Utilities.Utilities import Utilities


class ResNetImprovedV2(nn.Module):
    """
    Extended ResNet classifier for 2-D audio spectrograms.

    Builds on the baseline ResNet from the course by optionally inserting
    two attention modules controlled via boolean flags in hyperparameters,
    and by adding the progressive pooling strategy validated in a teammate's
    revised baseline ResNet:

        "use_se_block"     (bool) : if True, a SEBlock is appended after each
                                    pair of BasicResNetBlocks to perform
                                    channel-wise feature recalibration.

        "use_tf_attention" (bool) : if True, a TFAttentionBlock is inserted
                                    before the final pooling layer to apply
                                    joint time-frequency spatial attention.

        "pool_every_stage"  (bool) : if True, a MaxPool2d(2,2) is appended
                                    after each of the 4 residual stages,
                                    progressively halving F and T. This keeps
                                    feature map memory bounded in deeper
                                    layers (critical for the large spectrograms
                                    produced by N_MELS=128) and reduces the
                                    parameter count of the final LazyLinear,
                                    which in turn reduces overfitting risk.
                                    Default False, matching the original
                                    architecture (no intermediate pooling).

        "pool_output_size"  (tuple or None) : controls the final pooling layer
                                    before the classification head.
                                    If None (default): nn.AvgPool2d(kernel_size=(2,2)),
                                    matching the original behaviour.
                                    If a tuple (h, w): nn.AdaptiveAvgPool2d((h, w)),
                                    giving direct control over the Flatten size
                                    and therefore the LazyLinear parameter count.
                                    E.g. (1, 1) for a fully global pool (minimal
                                    parameters, used in early diagnostics), or
                                    (4, 4) as a compromise retaining some spatial
                                    structure while still controlling capacity.

    All remaining hyperparameters are identical to the baseline ResNet:
        input_dim, output_dim, hidden_layers_size, activation, kernel_size,
        filters, batch_normalization, dropout_rate.

    The network accepts tensors of shape (B, C, F, T) and returns logits
    of shape (B, output_dim), making it a drop-in replacement for ResNet
    with any existing Trainer and DataLoader.
    """

    def __init__(self, hyperparameters):
        nn.Module.__init__(self)

        self.hidden_layers_size  = hyperparameters["hidden_layers_size"]
        self.activation          = hyperparameters["activation"]
        self.batch_normalization = hyperparameters["batch_normalization"]
        self.dropout_rate        = hyperparameters["dropout_rate"]
        self.input_dim           = hyperparameters["input_dim"]
        self.output_dim          = hyperparameters["output_dim"]
        self.kernel_size         = hyperparameters["kernel_size"]
        self.filters              = hyperparameters["filters"]

        use_se_block       = hyperparameters.get("use_se_block",       False)
        use_tf_attention   = hyperparameters.get("use_tf_attention",   False)
        self.pool_every_stage = hyperparameters.get("pool_every_stage", False)
        self.pool_output_size = hyperparameters.get("pool_output_size", None)

        act = Utilities.get_activation(self.activation)

        self.layers = nn.ModuleList()

        # Initial projection: map input channels to the first filter bank.
        self.layers.append(
            Conv2DBlock(in_channels=self.input_dim[0],
                        out_channels=self.filters[0],
                        kernel_size=self.kernel_size,
                        activation=act,
                        batch_normalization=self.batch_normalization,
                        dropout_rate=self.dropout_rate)
        )

        # Residual backbone: 4 stages, each with a transition block (in -> out)
        # followed by a stabilisation block (out -> out)
        for i in range(4):
            in_ch  = self.filters[i]
            out_ch = self.filters[i + 1]

            self.layers.append(
                BasicResNetBlock(in_channels=in_ch, out_channels=out_ch,
                                 kernel_size=self.kernel_size, activation=act,
                                 batch_normalization=self.batch_normalization,
                                 dropout_rate=self.dropout_rate)
            )
            self.layers.append(
                BasicResNetBlock(in_channels=out_ch, out_channels=out_ch,
                                 kernel_size=self.kernel_size, activation=act,
                                 batch_normalization=self.batch_normalization,
                                 dropout_rate=self.dropout_rate)
            )

            # Optional SE recalibration after each residual stage.
            if use_se_block:
                self.layers.append(SEBlock(channels=out_ch))

            # Optional progressive spatial downsampling. Halves F and T after
            # each stage, keeping deep-layer feature maps small in memory and
            # bounding the parameter count of the eventual Flatten + LazyLinear.
            if self.pool_every_stage:
                self.layers.append(nn.MaxPool2d(kernel_size=(2, 2)))

        # Optional time-frequency spatial attention before global pooling.
        if use_tf_attention:
            self.layers.append(TFAttentionBlock())

        # Final pooling before the classification head.
        if self.pool_output_size is None:
            self.layers.append(nn.AvgPool2d(kernel_size=(2, 2)))
        else:
            self.layers.append(nn.AdaptiveAvgPool2d(self.pool_output_size))

        self.layers.append(nn.Flatten())
        self.layers.append(nn.LazyLinear(out_features=self.hidden_layers_size[0]))
        self.layers.append(
            DenseBlock(in_size=self.hidden_layers_size[0],
                       out_size=self.hidden_layers_size[1],
                       activation=act,
                       batch_normalization=self.batch_normalization,
                       dropout_rate=self.dropout_rate)
        )
        self.layers.append(
            nn.Linear(in_features=self.hidden_layers_size[1],
                      out_features=self.output_dim)
        )

        self.classifier = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.classifier(x)
