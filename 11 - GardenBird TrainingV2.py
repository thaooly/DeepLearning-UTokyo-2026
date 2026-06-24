import os
import json
from datetime import datetime

import torch
import numpy as np
from torchinfo import summary

from ClassesData.DatasetLoaderSpectrogram import DatasetLoader2D
from ClassesML.ResNetImproved import ResNetImproved
from ClassesML.ConvAutoEncoder import ConvAutoEncoder
from ClassesML.POCS import pocs_postprocess
from ClassesML.Scope import ScopeClassifier
from ClassesML.TrainerClassifier import TrainerClassifier


# ======================================================================
# AutoEncoder pre-training
# ======================================================================


def pretrain_autoencoder(x_train, hyperparameters, device):
    """
    Pre-train a ConvAutoEncoder on the training spectrograms using an
    unsupervised reconstruction objective (MSE loss on spectrogram space).

    Args:
        x_train         : list of spectrogram batches — list of Tensor (B, 1, F, T).
        hyperparameters : global hyperparameter dict; AE-specific keys:
                            "ae_base_channels"  — first encoder stage channel count (default: 16).
                            "ae_learning_rate"  — Adam learning rate (default: 1e-3).
                            "ae_max_epoch"      — number of pre-training epochs (default: 20).
        device          : torch.device used for training.

    Returns:
        Trained ConvAutoEncoder in eval() mode, on the specified device.
    """
    print("\n[ConvAutoEncoder] Starting unsupervised pre-training...")

    ae_model = ConvAutoEncoder(hyperparameters).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        ae_model.parameters(), lr=hyperparameters.get("ae_learning_rate", 1e-3)
    )

    n_epochs = hyperparameters.get("ae_max_epoch", 20)
    n_batch = len(x_train)

    for epoch in range(n_epochs):
        ae_model.train()
        total_loss = 0.0
        for n in range(n_batch):
            x = x_train[n].to(device)
            x_hat = ae_model(x)
            loss = criterion(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  AE Epoch {epoch + 1}/{n_epochs}  |  Loss: {total_loss / n_batch:.4f}")

    ae_model.eval()
    print("[ConvAutoEncoder] Pre-training complete.\n")
    return ae_model


# ======================================================================
# Preprocessing
# ======================================================================


def apply_preprocessing(x_batches, ae_model, hyperparameters, device):
    """
    Apply the optional AutoEncoder and/or POCS post-processing to a list of
    spectrogram batches.

    Processing order when both are enabled:
        AE reconstruction → POCS projection

    The AE removes out-of-manifold noise; POCS then enforces physical
    constraints (non-negativity, band-limitation, finite support) that the
    AE decoder may have violated during reconstruction.

    Args:
        x_batches       : list of Tensor (B, 1, F, T) on CPU.
        ae_model        : trained AutoEncoder or None if use_autoencoder=False.
        hyperparameters : global hyperparameter dict; flags "use_autoencoder"
                          and "use_pocs" control which steps are active.
        device          : torch.device for inference.

    Returns:
        List of processed Tensor (B, 1, F, T) on CPU.
    """
    use_ae = hyperparameters.get("use_autoencoder", False)
    use_pocs = hyperparameters.get("use_pocs", False)

    if not use_ae and not use_pocs:
        return x_batches

    processed = []
    with torch.no_grad():
        for x in x_batches:
            x = x.to(device)
            if use_ae:
                x = ae_model(x)
            if use_pocs:
                x = pocs_postprocess(x, hyperparameters)
            processed.append(x.cpu())
    return processed


# ======================================================================
# Main training
# ======================================================================


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    path_parent_project = os.getcwd()
    dataset_audio_path = os.path.join(
        path_parent_project, "Dataset", "mygardenbird_spectrogram_pt", ""
    )

    # ------------------------------------------------------------------
    # Hyperparameters
    # Toggle the boolean flags below to switch between experimental variants
    # without modifying any other part of the codebase. Each run is saved
    # under a unique timestamp in checkpoints/ for reproducible comparison.
    # ------------------------------------------------------------------
    hyperparameters = dict(
        # AutoEncoder denoising pre-stage
        # True  : pre-train an AE on training spectrograms, then filter all
        #         splits through it before the classifier sees them.
        # False : spectrograms are passed directly to the classifier.
        use_autoencoder=True,
        # POCS post-processing
        # True  : apply alternating projections (C1/C2/C3) after the AE
        #         (or directly on raw spectrograms if use_autoencoder=False).
        # False : no post-processing.
        use_pocs=True,
        # SE recalibration blocks inside ResNetImproved
        # True  : insert a SEBlock after each pair of BasicResNetBlocks.
        use_se_block=True,
        # Time-frequency attention inside ResNetImproved
        # True  : insert a TFAttentionBlock before the global average pool.
        use_tf_attention=True,
        # ResNet backbone.
        hidden_layers_size=[64, 128],
        activation="relu",
        kernel_size=(5, 5),
        filters=[4, 8, 16, 16, 16],
        batch_normalization=True,
        dropout_rate=0.01,
        learning_rate=1e-3,
        max_epoch=20,
        # ConvAutoEncoder architecture (used only if use_autoencoder=True).
        ae_base_channels=16,
        ae_learning_rate=1e-3,
        ae_max_epoch=20,
        # POCS projection parameters (used only if use_pocs=True).
        # Frequency bin indices for the band-pass mask (C2 projection).
        # At N_FFT=512 and sr=16 kHz, frequency resolution ≈ 31.25 Hz/bin:
        #   pocs_f_min_bin=2  → ~62 Hz lower guard (removes DC artefacts)
        #   pocs_f_max_bin=128 → upper bound within 128 mel bins
        pocs_n_iter=10,
        pocs_f_min_bin=2,
        pocs_f_max_bin=128,
    )

    # ------------------------------------------------------------------
    # 1. Data loading
    # ------------------------------------------------------------------
    print("\n[Data] Loading dataset...")
    dataset = DatasetLoader2D(root=dataset_audio_path, batch_size=32)
    train_dataset, val_dataset, input_dim, n_classes = (
        dataset.load_spectrogram_labels_data()
    )

    hyperparameters["input_dim"] = input_dim
    hyperparameters["output_dim"] = n_classes

    print(f"  Input shape : {input_dim}")
    print(f"  Classes     : {n_classes}  →  {dataset.classes}")

    # Full training :
    x_train = train_dataset[0]
    y_train = train_dataset[1]
    x_valid = val_dataset[0]
    y_valid = val_dataset[1]

    # Quick-test mode:
    # x_train = train_dataset[0][:2]
    # y_train = train_dataset[1][:2]
    # x_valid = val_dataset[0][:2]
    # y_valid = val_dataset[1][:2]

    # ------------------------------------------------------------------
    # 2. AutoEncoder pre-training (optional)
    # ------------------------------------------------------------------
    ae_model = None
    if hyperparameters["use_autoencoder"]:
        ae_model = pretrain_autoencoder(x_train, hyperparameters, device)

    # ------------------------------------------------------------------
    # 3. Preprocessing — AE reconstruction and/or POCS projection
    # ------------------------------------------------------------------
    print("[Preprocessing] Applying AE / POCS pipeline...")
    x_train = apply_preprocessing(x_train, ae_model, hyperparameters, device)
    x_valid = apply_preprocessing(x_valid, ae_model, hyperparameters, device)
    print("[Preprocessing] Done.")

    # ------------------------------------------------------------------
    # 4. Classifier instantiation
    # ------------------------------------------------------------------
    model = ResNetImproved(hyperparameters).to(device)
    scope = ScopeClassifier(model, hyperparameters)

    batch_size = x_train[0].shape[0]
    input_size = (batch_size, input_dim[0], input_dim[1], input_dim[2])
    print("\n[Model] Architecture summary:")
    print(
        summary(model=model, input_data=torch.rand(input_size, device=device), depth=5)
    )
    print("=" * 60)

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    print("\n[Training] Starting classifier training...")
    trainer = TrainerClassifier(hyperparameter=hyperparameters)
    trainer.set_model(model=model, device=device)
    trainer.set_scope(scope=scope)
    trainer.set_data(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid)

    train_accuracy_list, valid_accuracy_list = trainer.run()

    # ------------------------------------------------------------------
    # 6. Checkpoint saving
    # A single timestamp ties together all artefacts from this run:
    #   - classifier weights
    #   - AutoEncoder weights (if trained)
    #   - training metrics and hyperparameters (JSON)
    # The evaluation script uses this timestamp to reload the exact run.
    # ------------------------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    classifier_path = f"checkpoints/resnet_improved_{timestamp}.pt"
    ae_path = f"checkpoints/autoencoder_{timestamp}.pt"
    metrics_path = f"checkpoints/metrics_improved_{timestamp}.json"

    torch.save(model.state_dict(), classifier_path)

    if ae_model is not None:
        torch.save(ae_model.state_dict(), ae_path)
        print(f"[Checkpoint] AutoEncoder weights saved: {ae_path}")

    # Tuples are not JSON-serialisable; convert to lists before dumping.
    serialisable_hp = {
        k: (list(v) if isinstance(v, tuple) else v) for k, v in hyperparameters.items()
    }

    logs = {
        "hyperparameter": serialisable_hp,
        "train_accuracy_list": train_accuracy_list,
        "valid_accuracy_list": valid_accuracy_list,
        "classes": dataset.classes,
        "ae_weights_path": ae_path if ae_model is not None else None,
    }
    with open(metrics_path, "w") as f:
        json.dump(logs, f, indent=4)

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Classifier : {classifier_path}")
    if ae_model is not None:
        print(f"  AutoEncoder: {ae_path}")
    print(f"  Metrics    : {metrics_path}")
    print(
        f"  Flags      : AE={hyperparameters['use_autoencoder']}  "
        f"POCS={hyperparameters['use_pocs']}  "
        f"SE={hyperparameters['use_se_block']}  "
        f"TFA={hyperparameters['use_tf_attention']}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
