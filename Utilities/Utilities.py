import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
from torchvision.utils import make_grid

class Utilities:

    @staticmethod
    def get_activation(activation_str: str or None):

        if activation_str == 'relu':
            return nn.ReLU()
        elif activation_str == 'sigmoid':
            return nn.Sigmoid()
        elif activation_str == 'tanh':
            return nn.Tanh()
        elif activation_str == "linear":
            return None
        else:
            raise ValueError(f"Unknown activation function: {activation_str}")

    @staticmethod
    def images_as_canvas(images, title: str = ""):

        canvas = make_grid(images.cpu(), padding=10, nrow=10, normalize=True)
        canvas = canvas.permute(1, 2, 0).numpy() * 255
        canvas = canvas.astype("uint8")

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(canvas)
        ax.axis("off")
        ax.set_title(title)
        plt.show()

    @staticmethod
    def compute_accuracy(y, y_hat):

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        if not isinstance(y_hat, torch.Tensor):
            y_hat = torch.tensor(y_hat)

        _, predicted = torch.max(y_hat, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / y.size(0) * 100

        return accuracy

    @staticmethod
    def plot_confusion_matrix_fashion(y, y_hat):

        accuracy = Utilities.compute_accuracy(y, y_hat)

        y_hat = np.argmax(y_hat, 1)

        cm = confusion_matrix(y, y_hat)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        label_map = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover',
                     3: 'Dress', 4: 'Coat', 5: 'Sandal',
                     6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

        plt.figure()
        plt.subplot(1, 1, 1)
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=[label_map[i] for i in range(10)],
                    yticklabels=[label_map[i] for i in range(10)])
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion matrix - Accuracy: " + str(accuracy))
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_latent_space(z_fit, y_fit):

        label_map = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover',
                     3: 'Dress', 4: 'Coat', 5: 'Sandal',
                     6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(1, 1, 1)

        cmap = plt.get_cmap('gist_rainbow')
        colors = cmap(np.linspace(0, 1, 10))
        colors = dict(zip(label_map.keys(), colors))

        for y in label_map.keys():
            index = np.where(y_fit == y)
            ax.scatter(z_fit[index, 0], z_fit[index, 1], color=colors[y],
                       marker='o', s=30, alpha=0.5,
                       label=label_map[y])

        ax.legend()
        plt.show()

    @staticmethod
    def images_2_as_canvas(images, images2, title: str = ""):

        canvas = make_grid(images.cpu(), padding=10, nrow=10, normalize=True)
        canvas = canvas.permute(1, 2, 0).numpy() * 255
        canvas = canvas.astype("uint8")

        canvas2 = make_grid(images2.cpu(), padding=10, nrow=10, normalize=True)
        canvas2 = canvas2.permute(1, 2, 0).numpy() * 255
        canvas2 = canvas2.astype("uint8")

        canvas = np.concatenate((canvas, canvas2), axis=1)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(canvas)
        ax.axis("off")
        ax.set_title(title)
        plt.show()