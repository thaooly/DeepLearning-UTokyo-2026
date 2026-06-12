import torch
import torch.nn as nn

class RNN(nn.Module):

    def __init__(self, hyperparameters):

        super(RNN, self).__init__()

        self.hyper_model = hyperparameters

        # Extracting hyperparameters
        self.activation = self.hyper_model["activation"]
        self.input_dim = self.hyper_model["input_dim"]
        self.output_dim = self.hyper_model["output_dim"]
        self.rnn_hidden_size = self.hyper_model["rnn_hidden_size"]
        self.num_layers = self.hyper_model["num_layers"]
        self.dropout_rate = self.hyper_model["dropout_rate"]

        if "n_token" in self.hyper_model and "embedding_dim" in self.hyper_model:
            self.n_token = self.hyper_model["n_token"]
            self.embedding_dim = self.hyper_model["embedding_dim"]
            # Embedding
            self.embedding = nn.Embedding(self.n_token, self.embedding_dim)
        else:
            self.embedding = None

        self.rnn = nn.RNN(input_size=self.input_dim[0],
                          hidden_size=self.rnn_hidden_size[0],
                          num_layers=self.num_layers,
                          batch_first=True,
                          nonlinearity=self.activation.lower(),
                          dropout=self.dropout_rate)

        self.fc = nn.Linear(self.rnn_hidden_size[0], self.output_dim)

    def forward(self, x):
        if self.embedding:
            x = self.embedding(x)
        else:
            batch_size, channels, height, width = x.size()
            x = x.permute(0,2,3,1) # Change shape to [B, H, W, C]
            x = x.view(batch_size, height, width * channels) # Reshape to [B, H, W, C]

        # Hidden state
        h0 = torch.zeros((self.num_layers, x.size(0),
                         self.rnn_hidden_size[0])).to(x.device)

        out, _ = self.rnn(x, h0)
        x = out[:, -1, :]
        out = self.fc(x) # Use the last time step output for classification
        return out