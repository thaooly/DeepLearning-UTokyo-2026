import torch
import torch.nn as nn
from ClassesML.Blocks import DenseBlock, Conv2DBlock, BasicResNetBlock, SEBlock, TFAttentionBlock
from Utilities.Utilities import Utilities


class ResNetImproved(nn.Module):
    """
    Extended ResNet classifier for 2-D audio spectrograms.

    Two new modules controlled via boolean flags in hyperparameters 
    can be inserted into the baseline ResNet architecture:

        "use_se_block"     (bool) : if True, a SEBlock is appended after each
                                    pair of BasicResNetBlocks to perform
                                    channel-wise feature recalibration.

        "use_tf_attention" (bool) : if True, a TFAttentionBlock is inserted
                                    before the average pooling layer to apply
                                    joint time-frequency spatial attention.

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
        self.filters             = hyperparameters["filters"]

        use_se_block     = hyperparameters.get("use_se_block",     False)
        use_tf_attention = hyperparameters.get("use_tf_attention", False)

        act = Utilities.get_activation(self.activation)

        self.layers = nn.ModuleList()

        layer = Conv2DBlock(in_channels=self.input_dim[0],
                        out_channels=self.filters[0],
                        kernel_size=self.kernel_size,
                        activation=act)
        self.layers.append(layer)


        for i in range(4):
            in_ch  = self.filters[i]
            out_ch = self.filters[i + 1]

            layer = BasicResNetBlock(in_channels=in_ch, out_channels=out_ch,
                                 kernel_size=self.kernel_size, activation=act)
            self.layers.append(layer)
            layer = BasicResNetBlock(in_channels=out_ch, out_channels=out_ch,
                                 kernel_size=self.kernel_size, activation=act)
            self.layers.append(layer)

            # Optional SE recalibration after each residual stage.
            if use_se_block:
                self.layers.append(SEBlock(channels=out_ch))

        # Optional time-frequency spatial attention before global pooling.
        if use_tf_attention:
            self.layers.append(TFAttentionBlock())

        layer = nn.AvgPool2d(kernel_size=(2, 2))
        self.layers.append(layer)
        layer = nn.Flatten()
        self.layers.append(layer)
        layer = nn.LazyLinear(out_features=self.hidden_layers_size[0])
        self.layers.append(layer)
        layer = DenseBlock(in_size=self.hidden_layers_size[0],
                       out_size=self.hidden_layers_size[1],
                       activation=act,
                       batch_normalization=self.batch_normalization,
                       dropout_rate=self.dropout_rate)
        self.layers.append(layer)
        layer = nn.Linear(in_features=self.hidden_layers_size[1],
                      out_features=self.output_dim)
        self.layers.append(layer)

        self.classifier = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.classifier(x)
