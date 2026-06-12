import os 
import numpy as np
import pandas as pd
from tabulate import tabulate


import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from ClassesData.DatasetLoader import DatasetLoader
from ClassesML.CNN import CNN
from ClassesML.Scope import ScopeClassifier
from ClassesML.TrainerClassifier import TrainerClassifier
from ClassesML.SaveLoad import SaveLoad

import torch
import torch.optim as optim
import torch.nn as nn
from torchinfo import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_parent_project = os.getcwd()
dataset_image_path = os.path.join(path_parent_project, "Dataset", "FASHION", "")

dataset = DatasetLoader(root=dataset_image_path)
train_dataset, val_dataset, input_dim, n_classes = dataset.load_images_labels_data()




hyperparameters = dict(input_dim = input_dim,
                       output_dim = n_classes,
                       hidden_layers_size=[16,32],
                       activation="relu",
                       kernel_size = (5,5),
                       filters=[8,16,32],
                       batch_normalization =False,
                       dropout_rate = 0.1,
                       learning_rate = 0.001,
                       early_stopping = True,
                       patience_lr = 10,
                       max_epoch = 10)#push it to 100

from sklearn.model_selection import ParameterSampler

hyperparameters_choices={}

for k in hyperparameters.keys():
    hyperparameters_choices[k]=[hyperparameters[k]]

hyperparameters_choices["learning_rate"] = [0.01,0.005,0.001,0.0005,0.0001]
hyperparameters_choices["activation"] = ["relu","sigmoid","tanh"]
hyperparameters_choices["filters"] = [[8,16,32],[16,32,64]]

hyperparameters_try = list(ParameterSampler(hyperparameters_choices,n_iter=1))
metric_list=[]

for hyperparameters in hyperparameters_try:
    
    model = CNN(hyperparameters).to(device)
    scope = ScopeClassifier(model,hyperparameters)

    input_size = (128,hyperparameters["input_dim"][0],
                hyperparameters["input_dim"][1],
                hyperparameters["input_dim"][2])
    input_data = torch.rand(size=input_size,device=device)
    print(summary(model=model,input_data=input_data,depth=5))
    
    x_train = train_dataset[0]
    y_train = train_dataset[1]
    x_valid = val_dataset[0]
    y_valid = val_dataset[1]

    trainer = TrainerClassifier(hyperparameter=hyperparameters)
    trainer.set_model(model=model,device=device)
    trainer.set_scope(scope=scope)
    trainer.set_data(x_train=x_train,y_train=y_train,x_valid=x_valid,y_valid=y_valid)


    train_accuracy_list,valid_accuracy_list = trainer.run()

    metric_list.append(valid_accuracy_list[-1])
    hyperparameters["metric"] = valid_accuracy_list[-1]

#save load model
cwd = os.getcwd()
path_model = os.path.join(cwd,"Models")
model_name = "CNN_"
if not os.path.exists(path_model):
    os.makedirs(path_model)

save_load = SaveLoad(path_model=path_model,model_name=model_name)
save_load.save_model(model=model)

model = CNN(hyperparameters).to(device)#gives a network with random parameters on which we are going to save parameters
save_load = SaveLoad(path_model=path_model,model_name=model_name)
model = save_load.load_model(model=model)

idx = np.argsort(metric_list)
hyperparameters_sorted = np.array(hyperparameters_try)[idx].tolist()

df = pd.DataFrame.from_dict(hyperparameters_sorted)
print(tabulate(df,headers="keys",tablefmt="psql"))