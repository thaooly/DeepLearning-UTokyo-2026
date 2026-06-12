import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from ClassesData.DatasetLoader import DatasetLoader
from ClassesML.RNN import RNN
from ClassesML.Scope import ScopeClassifier
from ClassesML.TrainerClassifier import TrainerClassifier
from Utilities.Utilities import Utilities

import torch
import torch.optim as optim
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_parent_project = os.getcwd()
dataset_image_path = os.path.join(path_parent_project, "Dataset", "FASHION", "")
dataset = DatasetLoader(root=dataset_image_path)
train_dataset, val_dataset, input_dim, n_classes = dataset.load_images_labels_data()

hyperparameter = dict(input_dim=(input_dim[1], input_dim[0] * input_dim[2]),
                      output_dim=n_classes,
                      rnn_hidden_size=[256],
                      dropout_rate=0.05,
                      num_layers=2,
                      activation="tanh",
                      learning_rate=0.0001,
                      max_epoch=50)

model = RNN(hyperparameter).to(device)
scope = ScopeClassifier(model, hyperparameter)

input_size = (128, input_dim[0],input_dim[1],input_dim[2])
input_data = torch.rand(size=input_size, device=device)
print(summary(model=model, input_data=input_data, depth=5))

# Training
x_train = train_dataset[0]
y_train = train_dataset[1]
x_valid = val_dataset[0]
y_valid = val_dataset[1]

trainer = TrainerClassifier(hyperparameter=hyperparameter)
trainer.set_model(model=model, device=device)
trainer.set_scope(scope=scope)
trainer.set_data(x_train=x_train, y_train=y_train,
                 x_valid=x_valid, y_valid=y_valid)
train_accuracy_list, valid_accuracy_list = trainer.run()