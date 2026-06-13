import os
import json
from datetime import datetime
import numpy as np
import torch
from torchinfo import summary
from ClassesData.DatasetLoaderSpectrogram import DatasetLoader2D
from ClassesML.ResNet import ResNet
from ClassesML.Scope import ScopeClassifier
from ClassesML.TrainerClassifier import TrainerClassifier


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    path_parent_project = os.getcwd()
    dataset_audio_path = os.path.join(path_parent_project, "Dataset", "mygardenbird_ogg", "")

    # 1. Load Data Splits (Train & Val)
    dataset = DatasetLoader2D(root=dataset_audio_path, batch_size=32)
    train_dataset, val_dataset, input_dim, n_classes = dataset.load_spectrogram_labels_data()

    """x_train = train_dataset[0]
        y_train = train_dataset[1]
        x_valid = val_dataset[0]
        y_valid = val_dataset[1]"""

    # Fast testing constraints (Remove or comment out when doing your final, full training runs)
    x_train, y_train = train_dataset[0][:2], train_dataset[1][:2]
    x_valid, y_valid = val_dataset[0][:2], val_dataset[1][:2]

    # Verify basic metadata
    print("=" * 50)
    print("New Spectrogram Shape (Channels, Height, Width): " + str(input_dim))
    print("Number of classes: " + str(n_classes))
    print("Classes: " + str(dataset.classes))
    print("=" * 50)

    # 2. Hyperparameter Configuration
    hyperparameter = dict(
        input_dim=input_dim,
        output_dim=n_classes,
        hidden_layers_size=[64, 128],
        activation="relu",
        kernel_size=(5, 5),
        filters=[4, 8, 16, 16, 16],
        batch_normalization=True,
        dropout_rate=0.01,
        learning_rate=0.001,
        max_epoch=20
    )

    # 3. Instantiate Architecture & Setup Scope Environment
    model = ResNet(hyperparameter).to(device)
    scope = ScopeClassifier(model, hyperparameter)

    # 4. Print Architecture Summary Verification
    current_batch_size = x_train[0].shape[0]
    input_size = (
        current_batch_size,
        hyperparameter['input_dim'][0],
        hyperparameter["input_dim"][1],
        hyperparameter["input_dim"][2]
    )

    # Generate mock tensor to run forward diagnostic pass
    input_data = torch.rand(size=input_size, device=device)
    print("\n--- Model Summary ---")
    print(summary(model=model, input_data=input_data, depth=5))
    print("=" * 50)

    # 5. Train the Model using your exact working Trainer pattern
    print("\nStarting 2D Training Pipeline...")
    trainer = TrainerClassifier(hyperparameter=hyperparameter)
    trainer.set_model(model=model, device=device)
    trainer.set_scope(scope=scope)
    trainer.set_data(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid)

    train_accuracy_list, valid_accuracy_list = trainer.run()

    # 6. Create Timestamped File Paths
    os.makedirs("checkpoints", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    weights_path = f"checkpoints/resnet_bird_{timestamp}.pt"
    metrics_path = f"checkpoints/metrics_{timestamp}.json"

    # 7. Save State Weights Weights
    torch.save(model.state_dict(), weights_path)

    # 8. Save History Logs & Structure Metadata
    logs = {
        "hyperparameter": hyperparameter,
        "train_accuracy_list": train_accuracy_list,
        "valid_accuracy_list": valid_accuracy_list,
        "classes": dataset.classes
    }
    with open(metrics_path, "w") as f:
        json.dump(logs, f, indent=4)

    print("\n" + "=" * 50)
    print(f"Training Pipeline Complete!")
    print(f"Weights Exported: {weights_path}")
    print(f"Logs Exported:    {metrics_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()