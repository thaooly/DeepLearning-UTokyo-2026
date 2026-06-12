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
from ClassesML.GAN import Discriminator, Generator
from ClassesML.Scope import ScopeGAN
from Utilities.Utilities import Utilities

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_parent_project = os.getcwd()
dataset_image_path = os.path.join(path_parent_project, "Dataset", "FASHION", "")
dataset = DatasetLoader(root=dataset_image_path)
train_dataset, val_dataset, input_dim, n_classes = dataset.load_images_labels_data()

# Define hyperparameters
hyperparameter = dict(input_dim=input_dim,
                      output_dim=n_classes,
                      activation='relu',
                      discriminator_activation='relu',
                      filters=(128, 64, 32, 16),
                      discriminator_filters=(16, 32, 64),
                      kernel_size=(5,5),
                      embedding_dim=64,
                      latent_dim=64,
                      dropout_rate=0.2,
                      batch_normalization=False,
                      learning_rate=0.00001,
                      max_epoch=5)

generator = Generator(hyperparameter).to(device)

discriminator = Discriminator(hyperparameter).to(device)
scope = ScopeGAN(generator, discriminator, hyperparameter)

noise = torch.randn(128, hyperparameter["latent_dim"], device=device)
labels = torch.randint(0, 10, (128,), device=device)
print(summary(generator, input_data=[noise, labels], depth=5))

input_size = (128, hyperparameter["input_dim"][0], hyperparameter["input_dim"][1],
              hyperparameter["input_dim"][2])
input_data = torch.randn(input_size, device=device)
labels = torch.randint(0, 10, (128,), device=device)
print(summary(discriminator, input_data=[input_data, labels], depth=5))

x_train = train_dataset[0]
y_train = train_dataset[1]
x_valid = val_dataset[0]
y_valid = val_dataset[1]
train_loss_dict = {}
val_loss_dict = {}
max_epoch = hyperparameter["max_epoch"]
for epoch in range(max_epoch):
    # Train
    generator.train()
    discriminator.train()
    total_disc_loss = 0.0
    total_gen_loss = 0.0
    n_batch = len(x_train)
    for n in range(n_batch):
        real_images = x_train[n].to(device)
        y = y_train[n].to(device)
        batch_size = real_images.size(0)
        size = (batch_size, 1)
        scope.optimizer_discriminator.zero_grad()

        # Train discriminator with real images
        y_real_labels = torch.ones(*size, device=device)
        discriminator_output = discriminator(real_images, y)
        disc_loss_real = scope.criterion_discriminator(discriminator_output, y_real_labels)
        disc_loss_real.backward()
        # Train discriminator with fake images
        y_fake_labels = torch.zeros(*size, device=device)
        noise = torch.randn(batch_size, hyperparameter["latent_dim"], device=device)
        labels = torch.randint(0, hyperparameter["output_dim"], (batch_size,), device=device)
        fake_images = generator(noise, labels)
        discriminator_output = discriminator(fake_images.detach(), labels)
        disc_loss_fake = scope.criterion_discriminator(discriminator_output, y_fake_labels)
        disc_loss_fake.backward()
        disc_loss = disc_loss_real + disc_loss_fake
        scope.optimizer_discriminator.step()

        # Train generator
        scope.optimizer_generator.zero_grad()
        noise = torch.randn(batch_size, hyperparameter["latent_dim"],
                           device=device)
        labels = torch.randint(0, hyperparameter["output_dim"],
                               (batch_size,), device=device)
        y_valid_labels = torch.ones(*size, device=device)

        generated_outputs = generator(noise, labels)
        discriminator_outputs = discriminator(generated_outputs, labels)
        gen_loss = scope.criterion_generator(discriminator_outputs, y_valid_labels)
        gen_loss.backward()
        scope.optimizer_generator.step()

        total_disc_loss += disc_loss.item()
        total_gen_loss += gen_loss.item()

    train_disc_loss = total_disc_loss / n_batch
    train_gen_loss = total_gen_loss / n_batch

    print(f'Epoch [{epoch + 1}/{max_epoch}] - Discriminator Loss: {train_disc_loss:.4f} '
          f'- Generator Loss: {train_gen_loss:.4f}')

generator.eval()
discriminator.eval()

# Generate new images
n_classes = hyperparameter["output_dim"]
num_images_per_class = 10
latent_dim = hyperparameter["latent_dim"]

# Create grid of images
images = []
for label in range(n_classes):
    # Generate random noise
    noise = torch.randn(num_images_per_class, latent_dim, device=device)
    # Generate labels
    labels = torch.tensor(label, device=device).unsqueeze(0).repeat(num_images_per_class)
    # Generate conditional images
    with torch.no_grad():
        generated_images = generator(noise, labels).cpu()
    # Append generated images to the list
    images.append(generated_images)
# Combine images for each label into a sinngle tensor
generated_images = torch.cat(images, dim=0)
Utilities.images_as_canvas(generated_images)