import os
import json

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

matplotlib.use("TkAgg")

from ClassesData.DatasetLoaderSpectrogram import DatasetLoader2D
from ClassesML.ResNetImproved import ResNetImproved
from ClassesML.ConvAutoEncoder import ConvAutoEncoder
from ClassesML.POCS import pocs_postprocess


# ======================================================================
# Preprocessing
# ======================================================================

def apply_preprocessing(x_batches, ae_model, hyperparameters, device):
    """
    Reproduce the exact preprocessing pipeline used during training.

    The evaluation must apply the same transformations
    (AE reconstruction and/or POCS projection) in the same order as
    training, otherwise the classifier receives a distribution shift
    relative to what it was trained on, invalidating the evaluation.

    Args:
        x_batches       : list of Tensor (B, 1, F, T) on CPU.
        ae_model        : loaded AutoEncoder in eval() mode, or None.
        hyperparameters : hyperparameter dict loaded from the checkpoint JSON.
        device          : torch.device for inference.

    Returns:
        List of processed Tensor (B, 1, F, T) on CPU.
    """
    use_ae   = hyperparameters.get("use_autoencoder", False)
    use_pocs = hyperparameters.get("use_pocs",        False)

    if not use_ae and not use_pocs:
        return x_batches

    processed = []
    with torch.no_grad():
        for x in x_batches:
            x = x.to(device)
            if use_ae and ae_model is not None:
                x = ae_model(x)
            if use_pocs:
                x = pocs_postprocess(x, hyperparameters)
            processed.append(x.cpu())
    return processed


# ======================================================================
# Main evaluation
# ======================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    # ------------------------------------------------------------------
    # Target run — paste the timestamp from the training output here.
    # All checkpoint files for this run share this timestamp as a suffix.
    # ------------------------------------------------------------------
    TARGET_TIMESTAMP = "20260614_134319"
    # ------------------------------------------------------------------

    classifier_path = f"checkpoints/resnet_improved_{TARGET_TIMESTAMP}.pt"
    ae_path         = f"checkpoints/autoencoder_{TARGET_TIMESTAMP}.pt"
    metrics_path    = f"checkpoints/metrics_improved_{TARGET_TIMESTAMP}.json"

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier weights not found: {classifier_path}")

    # ------------------------------------------------------------------
    # 1. Reload run metadata
    # ------------------------------------------------------------------
    with open(metrics_path, "r") as f:
        logs = json.load(f)

    hyperparameters = logs["hyperparameter"]
    classes         = logs["classes"]

    # JSON deserialises tuples as lists; restore the expected types.
    hyperparameters["input_dim"]   = tuple(hyperparameters["input_dim"])
    hyperparameters["kernel_size"] = tuple(hyperparameters["kernel_size"])

    print(f"\nRun flags: "
          f"AE={hyperparameters.get('use_autoencoder')}  "
          f"POCS={hyperparameters.get('use_pocs')}  "
          f"SE={hyperparameters.get('use_se_block')}  "
          f"TFA={hyperparameters.get('use_tf_attention')}")

    # ------------------------------------------------------------------
    # 2. Rebuild classifier and load trained weights
    # ------------------------------------------------------------------
    print("\n[Model] Reconstructing classifier architecture...")
    model = ResNetImproved(hyperparameters)
    model.load_state_dict(torch.load(classifier_path, map_location=device))
    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 3. Reload AutoEncoder weights (if this run used one)
    # ------------------------------------------------------------------
    ae_model = None
    if hyperparameters.get("use_autoencoder", False):
        if not os.path.exists(ae_path):
            raise FileNotFoundError(
                f"AutoEncoder weights not found: {ae_path}\n"
                f"Ensure the training script saved them at this path."
            )
        print("[ConvAutoEncoder] Reloading weights...")
        ae_model = ConvAutoEncoder(hyperparameters)
        ae_model.load_state_dict(torch.load(ae_path, map_location=device))
        ae_model.to(device)
        ae_model.eval()
        print("[ConvAutoEncoder] Weights loaded successfully.")

    # ------------------------------------------------------------------
    # 4. Load test split with the same representation as training
    # ------------------------------------------------------------------
    print("\n[Data] Loading test split...")
    path_parent_project = os.getcwd()
    dataset_audio_path  = os.path.join(path_parent_project, "Dataset", "mygardenbird_ogg", "")

    dataset   = DatasetLoader2D(root=dataset_audio_path, batch_size=32)
    test_data = dataset.load_test_spectrogram_labels_data()

    x_test = test_data[0]
    y_test = test_data[1]

    # ------------------------------------------------------------------
    # 5. Apply the same preprocessing as during training
    # ------------------------------------------------------------------
    x_test = apply_preprocessing(x_test, ae_model, hyperparameters, device)

    # ------------------------------------------------------------------
    # 6. Batch-wise inference — avoids OOM on large test sets
    # ------------------------------------------------------------------
    print("[Inference] Running forward passes on test set...")
    all_pred, all_true = [], []

    with torch.no_grad():
        for n, (x, y) in enumerate(zip(x_test, y_test)):
            logits = model(x.float().to(device))
            preds  = torch.argmax(logits, dim=1).cpu().numpy()
            labels = y.numpy() if isinstance(y, torch.Tensor) else np.array(y)

            all_pred.append(preds)
            all_true.append(labels)

            if (n + 1) % 5 == 0 or (n + 1) == len(x_test):
                print(f"  Processed batch {n + 1} / {len(x_test)}")

    pred_labels = np.concatenate(all_pred)
    true_labels = np.concatenate(all_true)

    os.makedirs("results", exist_ok=True)
    # ------------------------------------------------------------------
    # 7. Learning curves
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(logs["train_accuracy_list"], "b-", linewidth=2, label="Train Accuracy")
    plt.plot(logs["valid_accuracy_list"], "r-", linewidth=2, label="Validation Accuracy")
    plt.title(f"Learning Curves — {TARGET_TIMESTAMP}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/learning_curves_{TARGET_TIMESTAMP}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ------------------------------------------------------------------
    # 8. Confusion matrix on the unseen test set
    # ------------------------------------------------------------------
    print("[Evaluation] Generating confusion matrix...")
    cm   = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, xticks_rotation="vertical", cmap=plt.cm.Blues)
    plt.title(
        f"GardenBird Classification — Confusion Matrix\n"
        f"Run: {TARGET_TIMESTAMP}  |  "
        f"AE={hyperparameters.get('use_autoencoder')}  "
        f"POCS={hyperparameters.get('use_pocs')}  "
        f"SE={hyperparameters.get('use_se_block')}  "
        f"TFA={hyperparameters.get('use_tf_attention')}"
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f"results/confusion_matrix_{TARGET_TIMESTAMP}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ------------------------------------------------------------------
    # 9. Final test accuracy
    # ------------------------------------------------------------------
    final_acc = (pred_labels == true_labels).mean() * 100
    print("\n" + "=" * 60)
    print(f"Run            : {TARGET_TIMESTAMP}")
    print(f"Test Accuracy  : {final_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
