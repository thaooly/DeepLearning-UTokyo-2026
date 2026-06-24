import copy

import torch
from Utilities.Utilities import Utilities


def spec_augment(x, freq_mask_param=15, time_mask_param=30, n_freq_masks=2, n_time_masks=2):
    """
    Apply SpecAugment to a batch of spectrograms. (regulariser against overfitting)

    Randomly masks contiguous bands of frequency bins and contiguous
    intervals of time frames, setting them to zero.

    Applied at every call (i.e. every batch, every epoch) so the
    network sees a different masking pattern each time it encounters a
    given clip.

    Args:
        x                : Tensor (B, 1, F, T) — batch of spectrograms.
        freq_mask_param  : maximum width (in bins) of each frequency mask.
        time_mask_param  : maximum width (in frames) of each time mask.
        n_freq_masks     : number of frequency masks applied per batch.
        n_time_masks     : number of time masks applied per batch.

    Returns:
        Tensor (B, 1, F, T) — augmented spectrograms (new tensor, input untouched).
    """
    x = x.clone()
    F = x.shape[2]
    T = x.shape[3]

    for _ in range(n_freq_masks):
        f = torch.randint(0, freq_mask_param + 1, (1,)).item()
        if f == 0 or f >= F:
            continue
        f0 = torch.randint(0, F - f, (1,)).item()
        x[:, :, f0:f0 + f, :] = 0.0

    for _ in range(n_time_masks):
        t = torch.randint(0, time_mask_param + 1, (1,)).item()
        if t == 0 or t >= T:
            continue
        t0 = torch.randint(0, T - t, (1,)).item()
        x[:, :, :, t0:t0 + t] = 0.0

    return x


def mixup_batch(x, y, alpha=0.4):
    """
    Apply Mixup augmentation to a batch of spectrograms.

    Creates synthetic training examples by linearly interpolating between
    pairs of samples and their labels.

    For each batch:
        lambda ~ Beta(alpha, alpha)
        x_mixed = lambda * x + (1 - lambda) * x[shuffled]
        y_mixed is returned as (y, y_shuffled, lambda) for use in mixup_loss

    Args:
        x     : Tensor (B, 1, F, T) — batch of spectrograms.
        y     : Tensor (B,) — integer class labels.
        alpha : Beta distribution parameter controlling mix strength.
                Low alpha (e.g. 0.2) → lambda near 0 or 1 (weak mixing).
                Higher alpha (e.g. 0.4) → more balanced mixing.

    Returns:
        x_mixed  : Tensor (B, 1, F, T) — mixed spectrograms.
        y_a      : Tensor (B,) — original labels.
        y_b      : Tensor (B,) — shuffled labels.
        lam      : float — mixing coefficient in [0, 1].
    """
    lam = torch.distributions.Beta(alpha, alpha).sample().item()

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    x_mixed = lam * x + (1 - lam) * x[index]
    y_a = y
    y_b = y[index]

    return x_mixed, y_a, y_b, lam


def mixup_loss(criterion, y_hat, y_a, y_b, lam):
    """
    Compute the Mixup loss as a weighted combination of two cross-entropy terms.

    Args:
        criterion : loss function (CrossEntropyLoss instance).
        y_hat     : Tensor (B, n_classes) — model logits.
        y_a       : Tensor (B,) — original labels.
        y_b       : Tensor (B,) — shuffled labels.
        lam       : float — mixing coefficient.

    Returns:
        Scalar loss tensor.
    """
    return lam * criterion(y_hat, y_a) + (1 - lam) * criterion(y_hat, y_b)


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

        # SpecAugment configuration (disabled by default).
        use_spec_augment = self.hyperparameter.get("use_spec_augment", False)
        freq_mask_param  = self.hyperparameter.get("spec_augment_freq_mask", 15)
        time_mask_param  = self.hyperparameter.get("spec_augment_time_mask", 30)
        n_freq_masks     = self.hyperparameter.get("spec_augment_n_freq_masks", 2)
        n_time_masks     = self.hyperparameter.get("spec_augment_n_time_masks", 2)

        # Mixup configuration (disabled by default).
        use_mixup  = self.hyperparameter.get("use_mixup", False)
        mixup_alpha = self.hyperparameter.get("mixup_alpha", 0.4)

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

                # SpecAugment (masking in frequency and time).
                if use_spec_augment:
                    x = spec_augment(
                        x,
                        freq_mask_param=freq_mask_param,
                        time_mask_param=time_mask_param,
                        n_freq_masks=n_freq_masks,
                        n_time_masks=n_time_masks,
                    )

                # Mixup (linear interpolation between pairs of samples).
                if use_mixup:
                    x, y_a, y_b, lam = mixup_batch(x, y, alpha=mixup_alpha)
                    y_hat = self.model(x)
                    loss  = mixup_loss(self.scope.criterion, y_hat, y_a, y_b, lam)
                    # Accuracy is measured against the dominant label (y_a),
                    # which corresponds to the larger mixing coefficient lam.
                    # This slightly underestimates true performance when lam < 0.5
                    # but gives a consistent and interpretable training metric.
                    correct = self.count_correct(y_a, y_hat)
                else:
                    y_hat   = self.model(x)
                    loss    = self.scope.criterion(y_hat, y)
                    correct = self.count_correct(y, y_hat)

                # Backward propagation
                self.scope.optimizer.zero_grad()
                loss.backward()

                max_grad_norm = self.hyperparameter.get("max_grad_norm", None)
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                self.scope.optimizer.step()

                batch_size = y.size(0)
                total_train_loss    += loss.item() * batch_size
                total_train_correct += correct
                total_train_samples += batch_size
            
            avg_train_loss     = total_train_loss / total_train_samples
            avg_train_accuracy = total_train_correct / total_train_samples * 100

            # --- PHASE DE VALIDATION ---
            self.model.eval()
            total_valid_loss = 0.0
            total_valid_correct = 0
            total_valid_samples = 0
            n_batch_valid = len(self.x_valid)

            with torch.no_grad():
                for n in range(n_batch_valid):
                    x = self.x_valid[n].to(self.device)
                    y = self.y_valid[n].to(self.device)

                    y_hat = self.model(x)
                    loss  = self.scope.criterion(y_hat, y)

                    batch_size = y.size(0)
                    total_valid_loss    += loss.item() * batch_size
                    total_valid_correct += self.count_correct(y, y_hat)
                    total_valid_samples += batch_size
            
            avg_valid_loss     = total_valid_loss / total_valid_samples
            avg_valid_accuracy = total_valid_correct / total_valid_samples * 100
            
            # Affichage des logs
            print(f"Epoch: {epoch+1}/{self.hyperparameter['max_epoch']}")
            print(f"Training Loss: {avg_train_loss:.4f} - Training Acc: {avg_train_accuracy:.4f}")
            print(f"Validation Loss: {avg_valid_loss:.4f} - Validation Acc: {avg_valid_accuracy:.4f}")
            print("-" * 30)

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
                self.model, keep_training = self.scope.early_stopper.set(
                    model=self.model, epoch=epoch, metric_epoch=validation_metric
                )
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
        self.n_batches = 100  # if too slow, change, OG was 2000
