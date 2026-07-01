import torch.nn as nn
from ClassesML.Blocks import DenseBlock, Conv2DBlock, BasicResNetBlock, SEBlock, TFAttentionBlock
from Utilities.Utilities import Utilities


class ResNetImprovedV2(nn.Module):
    """
    Same backbone as the course ResNet, with optional add-ons turned on/off
    via flags in hyperparameters:

        use_se_block      : adds a SEBlock after each residual stage
        use_tf_attention   : adds a TFAttentionBlock before the final pooling
        pool_every_stage   : adds MaxPool2d(2,2) after each stage, halving F/T each time.
        pool_output_size   : controls the last pooling before Flatten.
                              None -> AvgPool2d(2,2) (default)
                              tuple (h,w) -> AdaptiveAvgPool2d((h,w))

    Everything else is the same as the baseline ResNet.
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

        # first conv, maps input channels to filters[0]
        self.layers.append(
            Conv2DBlock(in_channels=self.input_dim[0],
                        out_channels=self.filters[0],
                        kernel_size=self.kernel_size,
                        activation=act,
                        batch_normalization=self.batch_normalization,
                        dropout_rate=self.dropout_rate)
        )

        # 4 residual stages
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

            if use_se_block:
                self.layers.append(SEBlock(channels=out_ch))

            if self.pool_every_stage:
                self.layers.append(nn.MaxPool2d(kernel_size=(2, 2)))

        if use_tf_attention:
            self.layers.append(TFAttentionBlock())

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
