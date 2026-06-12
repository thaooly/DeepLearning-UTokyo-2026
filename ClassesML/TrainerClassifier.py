import torch
from Utilities.Utilities import Utilities

class TrainerClassifier:

    def __init__(self, hyperparameter):
        self.hyperparameter = hyperparameter
        self.model = None
        self.device = None
        self.scope = None
        self.x_train, self.y_train = None, None
        self.x_valid, self.y_valid = None, None
    
    def set_model(self, model, device):
        self.model = model
        self.device = device
    
    def set_scope(self, scope):
        self.scope = scope
    
    def set_data(self, x_train, y_train, x_valid, y_valid):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
    
    def run(self):
        train_accuracy_list = []
        valid_accuracy_list = []

        for epoch in range(self.hyperparameter["max_epoch"]):
            # --- PHASE D'ENTRAÎNEMENT ---
            self.model.train()
            total_train_loss = 0.0
            total_train_accuracy = 0.0
            n_batch_train = len(self.x_train)

            for n in range(n_batch_train):
                x = self.x_train[n].to(self.device)
                y = self.y_train[n].to(self.device)

                # Forward pass
                y_hat = self.model(x)
                loss = self.scope.criterion(y_hat, y)

                # Backward propagation (Mise à jour des poids)
                self.scope.optimizer.zero_grad()
                loss.backward()
                self.scope.optimizer.step()

                total_train_loss += loss.item()
                total_train_accuracy += Utilities.compute_accuracy(y, y_hat)
            
            avg_train_loss = total_train_loss / n_batch_train
            avg_train_accuracy = total_train_accuracy / n_batch_train

            # --- PHASE DE VALIDATION ---
            self.model.eval()
            total_valid_loss = 0.0
            total_valid_accuracy = 0.0
            n_batch_valid = len(self.x_valid)

            # Désactivation des gradients pour la validation
            with torch.no_grad():
                for n in range(n_batch_valid):
                    x = self.x_valid[n].to(self.device)
                    y = self.y_valid[n].to(self.device)

                    # Forward pass uniquement
                    y_hat = self.model(x)
                    loss = self.scope.criterion(y_hat, y)

                    total_valid_loss += loss.item()
                    total_valid_accuracy += Utilities.compute_accuracy(y, y_hat)
            
            avg_valid_loss = total_valid_loss / n_batch_valid
            avg_valid_accuracy = total_valid_accuracy / n_batch_valid
            
            # Affichage des logs
            print(f"Epoch: {epoch+1}/{self.hyperparameter['max_epoch']}")
            print(f"Training Loss: {avg_train_loss:.4f} - Training Acc: {avg_train_accuracy:.4f}")
            print(f"Validation Loss: {avg_valid_loss:.4f} - Validation Acc: {avg_valid_accuracy:.4f}")
            print("-" * 30)

            # Stockage des résultats

            if self.scope.scheduler:
                validation_metric = avg_valid_accuracy 
                old_lr = self.scope.optimizer.param_groups[0]['lr']
                self.scope.scheduler.step(validation_metric)
                new_lr = self.scope.optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    print(f'learning rate changed from {old_lr} to {new_lr} at epoch {epoch}')

            
            if self.scope.early_stopper:
                validation_metric = avg_valid_accuracy
                self.model, keep_training = self.scope.early_stopper.set(model=self.model,epoch=epoch,metric_epoch=validation_metric)
                if not keep_training:
                    break



            train_accuracy_list.append(avg_train_accuracy)
            valid_accuracy_list.append(avg_valid_accuracy)
        
        
        return train_accuracy_list, valid_accuracy_list

class TrainerTextClassifier:
    def __init__(self, hyperparameter):
        self.hyperparameter = hyperparameter
        self.n_batches = 100 # if too slow, change, OG was 2000