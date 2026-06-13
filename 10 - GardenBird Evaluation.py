import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ClassesData.DatasetLoaderSpectrogram import DatasetLoader2D
from ClassesML.ResNet import ResNet

matplotlib.use("TkAgg")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using evaluation device: {device}")

    # =====================================================================
    # PASTE SPECIFIC TIMESTAMP FILENAME HERE FROM YOUR CHECKPOINTS FOLDER
    # =====================================================================
    TARGET_TIMESTAMP = "20260613_214936"
    # =====================================================================

    metrics_path = f"checkpoints/metrics_{TARGET_TIMESTAMP}.json"
    weights_path = f"checkpoints/resnet_bird_{TARGET_TIMESTAMP}.pt"

    if not os.path.exists(metrics_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint data for timestamp '{TARGET_TIMESTAMP}' missing. Verify files exist.")

    # 1. Pull Up Historic Meta Parameters
    with open(metrics_path, "r") as f:
        logs = json.load(f)

    hyperparameter = logs["hyperparameter"]
    classes = logs["classes"]

    # 2. Re-instantiate Structural Shell & Inject Trained Matrices
    print("Reconstructing architecture and mapping state dictionaries...")
    model = ResNet(hyperparameter)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Fetch Final Examination Arrays (Test Split)
    print("Loading test split audio arrays...")
    path_parent_project = os.getcwd()
    dataset_audio_path = os.path.join(path_parent_project, "Dataset", "mygardenbird_ogg", "")
    dataset = DatasetLoader2D(root=dataset_audio_path, batch_size=32)

    test_dataset = dataset.load_test_spectrogram_labels_data()
    x_test_batches = test_dataset[0]
    y_test_batches = test_dataset[1]

    # 4. Safe Batch-by-Batch Inference
    print("Running forward evaluation passes in batches...")
    all_pred_labels = []
    all_true_labels = []

    with torch.no_grad():
        # Loop through the pre-packaged batches instead of slamming them together
        for n in range(len(x_test_batches)):
            # Get one small batch
            batch_x = x_test_batches[n].float().to(device)
            batch_y = y_test_batches[n]

            # Forward pass through the network
            raw_predictions = model(batch_x)
            pred_batch = torch.argmax(raw_predictions, dim=1).cpu().numpy()

            # Decode target labels if they are one-hot encoded
            if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
                batch_y = np.argmax(batch_y, axis=1)

            all_pred_labels.append(pred_batch)
            all_true_labels.append(batch_y)

            if (n + 1) % 5 == 0 or (n + 1) == len(x_test_batches):
                print(f"  -> Processed batch {n + 1}/{len(x_test_batches)}")

    # Safely stitch only the tiny scalar prediction results back together
    pred_labels = np.concatenate(all_pred_labels)
    y_test = np.concatenate(all_true_labels)

    # 5. Graphing Historical Progress Results Trace
    plt.figure(figsize=(10, 4))
    plt.plot(logs["train_accuracy_list"], "b-", linewidth=2, label="Train Accuracy")
    plt.plot(logs["valid_accuracy_list"], "r-", linewidth=2, label="Valid Accuracy")
    plt.title(f"Historical Learning Progress Trace ({TARGET_TIMESTAMP})")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()

    # 6. Build and Display Confusion Matrix Map
    print("Generating classification breakdown visual maps...")
    cm = confusion_matrix(y_test, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap=plt.cm.Blues)
    plt.title("GardenBird Audio 2D Classification Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Absolute Final Accuracy Grade
    final_acc = (np.sum(pred_labels == y_test) / len(y_test)) * 100
    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS FOR RUN: {TARGET_TIMESTAMP}")
    print(f"Final Unbiased Test Accuracy Score: {final_acc:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()