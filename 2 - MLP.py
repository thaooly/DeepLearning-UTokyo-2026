import os
import numpy as np

import matplotlib
from diffusers.pipelines.ltx.pipeline_ltx_i2v_long_multi_prompt import batch_normalize
from torch.ao.nn.quantized.modules import activation

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
from torchinfo import summary

from ClassesData.DatasetLoader import DatasetLoader
from Utilities.Utilities import Utilities
from ClassesML.MLP import MLP
from ClassesML.Scope import ScopeClassifier
from ClassesML.TrainerClassifier import TrainerClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_parent_project = os.getcwd()
dataset_image_path = os.path.join(path_parent_project, "Dataset", "FASHION", "")

dataset = DatasetLoader(root=dataset_image_path)
train_dataset, val_dataset, input_dim, n_classes = dataset.load_images_labels_data()

images = train_dataset[0][0].to(device)
labels = train_dataset[1][0].to(device)

Utilities.images_as_canvas(images=images)

hyperparameter = dict(input_dim=input_dim,
                      output_dim=n_classes,
                      hidden_layers_size=[64,128,256],
                      activation="relu",
                      batch_normalization=False,
                      dropout_rate=0.01,
                      learning_rate=0.001,
                      max_epoch=100)

model = MLP(hyperparameter).to(device)
scope = ScopeClassifier(model, hyperparameter)

input_size = (128, hyperparameter['input_dim'][0],
              hyperparameter["input_dim"][1],
              hyperparameter["input_dim"][2])

input_data = torch.rand(size=input_size, device=device)
print(summary(model=model, input_data=input_data, depth=5))

x_train = train_dataset[0]
y_train = train_dataset[1]
x_valid = val_dataset[0]
y_valid = val_dataset[1]

trainer = TrainerClassifier(hyperparameter=hyperparameter)
trainer.set_model(model=model, device=device)
trainer.set_scope(scope=scope)
trainer.set_data(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid)
train_accuracy_list, valid_accuracy_list = trainer.run()

plt.figure()
plt.plot(train_accuracy_list, "b-", label="Train Accuracy")
plt.plot(valid_accuracy_list, "r-", label="Valid Accuracy")
plt.show()

x = np.concatenate([x_valid[n] for n in range(len(x_valid))])
x = torch.from_numpy(x).to(device)
y= np.concatenate([y_valid[n] for n in range(len(y_valid))])
y = torch.from_numpy(y).to(device)

y_hat = model(x)

y_cpu = y.cpu().detach().numpy()
y_hat_cpu = y_hat.cpu().detach().numpy()

Utilities.plot_confusion_matrix_fashion(y_cpu, y_hat_cpu)
