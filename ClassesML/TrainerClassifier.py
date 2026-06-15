import copy

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
        best_valid_accuracy = -1.0
        best_epoch = 0
        best_model_weights = None

        for epoch in range(self.hyperparameter["max_epoch"]):
            # --- PHASE D'ENTRAÎNEMENT ---
            self.model.train()
            total_train_loss = 0.0
            total_train_correct = 0
            total_train_samples = 0
            n_batch_train = len(self.x_train)

            batch_indices = torch.randperm(n_batch_train).tolist()

            for n in batch_indices:
                x = self.x_train[n].to(self.device)
                y = self.y_train[n].to(self.device)

                # Forward pass
                y_hat = self.model(x)
                loss = self.scope.criterion(y_hat, y)

                # Backward propagation (Mise à jour des poids)
                self.scope.optimizer.zero_grad()
                loss.backward()

                max_grad_norm = self.hyperparameter.get("max_grad_norm", None)
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                self.scope.optimizer.step()

                batch_size = y.size(0)
                total_train_loss += loss.item() * batch_size
                total_train_correct += self.count_correct(y, y_hat)
                total_train_samples += batch_size
            
            avg_train_loss = total_train_loss / total_train_samples
            avg_train_accuracy = total_train_correct / total_train_samples * 100

            # --- PHASE DE VALIDATION ---
            self.model.eval()
            total_valid_loss = 0.0
            total_valid_correct = 0
            total_valid_samples = 0
            n_batch_valid = len(self.x_valid)

            # Désactivation des gradients pour la validation
            with torch.no_grad():
                for n in range(n_batch_valid):
                    x = self.x_valid[n].to(self.device)
                    y = self.y_valid[n].to(self.device)

                    # Forward pass uniquement
                    y_hat = self.model(x)
                    loss = self.scope.criterion(y_hat, y)

                    batch_size = y.size(0)
                    total_valid_loss += loss.item() * batch_size
                    total_valid_correct += self.count_correct(y, y_hat)
                    total_valid_samples += batch_size
            
            avg_valid_loss = total_valid_loss / total_valid_samples
            avg_valid_accuracy = total_valid_correct / total_valid_samples * 100
            
            # Affichage des logs
            print(f"Epoch: {epoch+1}/{self.hyperparameter['max_epoch']}")
            print(f"Training Loss: {avg_train_loss:.4f} - Training Acc: {avg_train_accuracy:.4f}")
            print(f"Validation Loss: {avg_valid_loss:.4f} - Validation Acc: {avg_valid_accuracy:.4f}")
            print("-" * 30)

            # Stockage des résultats

            if avg_valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = avg_valid_accuracy
                best_epoch = epoch + 1
                best_model_weights = copy.deepcopy(self.model.state_dict())
                print(f"Best Validation Acc: {best_valid_accuracy:.4f} - Keeping weights")

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

        if best_model_weights is not None and self.hyperparameter.get("restore_best_model", False):
            self.model.load_state_dict(best_model_weights)
            print(f"Restored best validation weights: epoch {best_epoch} - {best_valid_accuracy:.4f}")

        self.best_epoch = best_epoch
        self.best_valid_accuracy = best_valid_accuracy
        
        
        return train_accuracy_list, valid_accuracy_list

    def count_correct(self, y, y_hat):
        _, predicted = torch.max(y_hat, 1)
        return (predicted == y).sum().item()

class TrainerTextClassifier:
    def __init__(self, hyperparameter):
        self.hyperparameter = hyperparameter
        self.n_batches = 100 # if too slow, change, OG was 2000
