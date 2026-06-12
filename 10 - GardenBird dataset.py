import os

import torch

from ClassesData.GardenBirdDatasetLoader import GardenBirdDatasetLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_parent_project = os.getcwd()
dataset_audio_path = os.path.join(path_parent_project, "Dataset", "mygardenbird_ogg", "")

dataset = GardenBirdDatasetLoader(root=dataset_audio_path)

train_dataset, val_dataset, input_dim, n_classes = dataset.load_audio_labels_data()

x_train = train_dataset[0]
y_train = train_dataset[1]
x_valid = val_dataset[0]
y_valid = val_dataset[1]

print("Input dim: " + str(input_dim))
print("Number of classes: " + str(n_classes))
print("Classes: " + str(dataset.classes))

x = x_train[0].to(device)
y = y_train[0].to(device)

print("First batch X: " + str(x.shape))
print("First batch Y: " + str(y.shape))
