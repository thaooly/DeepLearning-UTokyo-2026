import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ClassesML.EarlyStopper import EarlyStopper

class ScopeClassifier:

    def __init__(self,model,hyperparameters):

        self.criterion = nn.CrossEntropyLoss()

        weight_decay = hyperparameters.get("weight_decay", 0.0)
        optimizer_name = hyperparameters.get("optimizer", "adam")

        if optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=hyperparameters["learning_rate"],
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=hyperparameters["learning_rate"],
                weight_decay=weight_decay,
            )

        if "patience_lr" in hyperparameters:
            lr_factor = hyperparameters.get("lr_factor", 0.1)
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=hyperparameters["patience_lr"],
                factor=lr_factor,
            )
        else:
            self.scheduler = None 

        if "early_stopping" in hyperparameters:
            self.early_stopper = EarlyStopper(hyperparameters=hyperparameters)
        else:
            self.early_stopper = None

class ScopeGAN:
    def __init__(self, generator, discriminator, hyperparameters):
        self.criterion_generator = nn.MSELoss()
        self.criterion_discriminator = nn.BCELoss()
        self.optimizer_generator = optim.Adam(generator.parameters(),
                                              lr=hyperparameters["learning_rate"])
        self.optimizer_discriminator = optim.Adam(discriminator.parameters(),
                                                  lr=hyperparameters["learning_rate"])

class ScopeAutoencoder:

    def __init__(self, model, hyperparameters):
        self.criterion = nn.MSELoss()

        encoder_parameters = list(model.encoder.parameters())
        decoder_parameters = list(model.decoder.parameters())
        autoencoder_parameters = encoder_parameters + decoder_parameters

        self.optimizer = optim.Adam(autoencoder_parameters, lr=hyperparameters["learning_rate"])
