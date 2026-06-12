import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import torch
from torchinfo import summary
from torch.utils.data import DataLoader, Dataset

from ClassesData.DatasetLoader import DatasetLoader, AGNewsDataset
from ClassesML.RNN import RNN
from ClassesML.Scope import ScopeClassifier
from ClassesML.TrainerClassifier import TrainerTextClassifier
from Utilities.Utilities import Utilities

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DatasetLoader(root="")
dataset, tokenizer = dataset.load_text_data() # tokenizer turns words into integer encoding

#  Define constants
embedding_dim = 64
seq_length = 32 # Maximum length of input tokens
n_classes = 4 # Number of classes in AG News dataset
n_token = len(tokenizer) # amount of words in the tokenizer dictionary
input_dim = (embedding_dim, seq_length)

# Create instances of training and validation datasets
train_dataset = AGNewsDataset(dataset['train'], tokenizer, seq_length)
test_dataset = AGNewsDataset(dataset['test'], tokenizer, seq_length)

hyperparameter = dict(input_dim=input_dim,
                      output_dim=n_classes,
                      n_token=n_token,
                      rnn_hidden_size=[256],
                      dropout_rate=0.01,
                      num_layers=2,
                      embedding_dim=embedding_dim,
                      activation="tanh",
                      learning_rate=0.001,
                      max_epoch=10)

model = RNN(hyperparameter).to(device)
scope = ScopeClassifier(model, hyperparameter)

input_size = (64, hyperparameter['input_dim'][1])
input_data = torch.randint(0, 10000, input_size, device=device, dtype=torch.int)
print(summary(model=model, input_data=input_data, depth=5))