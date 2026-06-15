import os

import torch


class GardenBirdDatasetLoader:
    def __init__(self, root):

        self.root = root
        self.classes = self.load_classes()

    def load_audio_labels_data(self):

        train_data_batches = torch.load(os.path.join(self.root, "train_data_batches.pt"))
        train_label_batches = torch.load(os.path.join(self.root, "train_label_batches.pt"))

        val_data_batches = torch.load(os.path.join(self.root, "val_data_batches.pt"))
        val_label_batches = torch.load(os.path.join(self.root, "val_label_batches.pt"))

        train_data_batches = [batch.float() for batch in train_data_batches]
        val_data_batches = [batch.float() for batch in val_data_batches]

        train_dataset = [train_data_batches, train_label_batches]
        val_dataset = [val_data_batches, val_label_batches]

        input_dim = train_dataset[0][0].numpy().shape[1:]
        n_classes = len(self.classes)

        return train_dataset, val_dataset, input_dim, n_classes

    def load_test_audio_labels_data(self):

        test_data_batches = torch.load(os.path.join(self.root, "test_data_batches.pt"))
        test_label_batches = torch.load(os.path.join(self.root, "test_label_batches.pt"))

        test_data_batches = [batch.float() for batch in test_data_batches]
        test_dataset = [test_data_batches, test_label_batches]

        return test_dataset

    def load_classes(self):

        classes_path = os.path.join(self.root, "classes.txt")

        with open(classes_path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f.readlines()]

        return classes
