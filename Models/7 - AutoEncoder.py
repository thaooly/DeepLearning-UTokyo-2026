import os
import numpy as np
import matplotlib
from rich import scope

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import copy
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import make_grid
from torchinfo import summary
from ClassesData.DatasetLoader import DatasetLoader
from ClassesML.AutoEncoder import AutoEncoder
from ClassesML.Scope import ScopeAutoencoder
from Utilities.Utilities import Utilities

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_parent_project = os.getcwd()
dataset_image_path = os.path.join(path_parent_project, "Dataset", "FASHION", "")
dataset = DatasetLoader(root=dataset_image_path)
train_dataset, val_dataset, input_dim, n_classes = dataset.load_images_labels_data()

hyperparameter = dict(input_dim=input_dim,
                      output_dim=n_classes,
                      output_activation="sigmoid",
                      latent_dim=[2],
                      hidden_layers_sizes=[512, 256, 128, 64],
                      activation="relu",
                      batch_normalization=False,
                      dropout_rate=0.05,
                      learning_rate=0.001,
                      max_epoch=50)

model = AutoEncoder(hyperparameter).to(device)
scope = ScopeAutoencoder(model, hyperparameter)

input_size = (128, hyperparameter["input_dim"][0],
              hyperparameter["input_dim"][1],
              hyperparameter["input_dim"][2])

input_data = torch.rand(size=input_size, device=device)
summary(model=model, input_data=input_data, depth=5)

# Training
x_train = train_dataset[0]
y_train = train_dataset[1]
x_valid = val_dataset[0]
y_valid = val_dataset[1]


# Train autoEncoder
train_loss_dict = {}
valid_loss_dict = {}

for epoch in range(hyperparameter["max_epoch"]):

    model.train()

    total_loss = 0.0
    n_batch = len(x_train)

    for n in range(n_batch):

        x = x_train[n].to(device)
        # forward pass
        x_hat = model(x)
        loss = scope.criterion(x_hat, x)
        # Backward propagation
        scope.optimizer.zero_grad()
        loss.backward()
        scope.optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / n_batch

    print("Epoch: " + str(epoch + 1) + "/" + str(hyperparameter["max_epoch"]))
    print("Training Loss: " + str(train_loss))

    #-------------------------------------------

    total_loss = 0.0
    n_batch = len(x_valid)
    model.eval()

    for n in range(n_batch):
        x = x_valid[n].to(device)
        # Forward pass
        x_hat = model(x)
        loss = scope.criterion(x_hat, x)
        total_loss += loss.item()

    valid_loss = total_loss / n_batch
    print("Validation Loss: " + str(valid_loss))

    train_loss_dict[epoch] = train_loss
    valid_loss_dict[epoch] = valid_loss

# Plot latent space
x_fit = torch.cat(train_dataset[0], dim=0).to(device)
y_fit = torch.cat(train_dataset[1], dim=0)

x_transform = torch.cat(val_dataset[0], dim=0).to(device)
y_transform = torch.cat(val_dataset[1], dim=0)

z_fit = model.encoder(x_fit).detach().cpu().numpy()
z_transform = model.encoder(x_transform).detach().cpu().numpy()

Utilities.plot_latent_space(z_fit, y_fit)
Utilities.plot_latent_space(z_transform, y_transform)

# Sample linearly to generate new sample along an axis and plot generated images
n_samples = 100
x_values = torch.zeros(n_samples, device=device)
y_values = torch.linspace(0, 10, n_samples, device=device)
z = torch.stack((x_values, y_values), dim=1)
generated_images = model.sample(z=z)
Utilities.images_as_canvas(generated_images)

# Plot original and reconstructed images
source_images = train_dataset[0][0].to(device)
reconstructed_images = model(source_images)
Utilities.images_2_as_canvas(source_images, reconstructed_images)