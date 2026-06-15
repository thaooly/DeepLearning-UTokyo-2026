import os

import torch


EXPECTED_INPUT_DIM = (1, 128, 301)


class DatasetLoader2D:
    def __init__(self, root, batch_size=32):

        self.root = root
        self.batch_size = batch_size
        self.classes = self.load_classes()

    def load_spectrogram_labels_data(self):

        train_data_batches = torch.load(os.path.join(self.root, "train_data_batches.pt"))
        train_label_batches = torch.load(os.path.join(self.root, "train_label_batches.pt"))

        val_data_batches = torch.load(os.path.join(self.root, "val_data_batches.pt"))
        val_label_batches = torch.load(os.path.join(self.root, "val_label_batches.pt"))

        train_data_batches = [batch.float() for batch in train_data_batches]
        val_data_batches = [batch.float() for batch in val_data_batches]

        self.check_spectrogram_batches(train_data_batches, train_label_batches, "train")
        self.check_spectrogram_batches(val_data_batches, val_label_batches, "val")

        train_dataset = [train_data_batches, train_label_batches]
        val_dataset = [val_data_batches, val_label_batches]

        input_dim = train_dataset[0][0].numpy().shape[1:]
        n_classes = len(self.classes)

        return train_dataset, val_dataset, input_dim, n_classes

    def load_test_spectrogram_labels_data(self):

        test_data_batches = torch.load(os.path.join(self.root, "test_data_batches.pt"))
        test_label_batches = torch.load(os.path.join(self.root, "test_label_batches.pt"))

        test_data_batches = [batch.float() for batch in test_data_batches]
        self.check_spectrogram_batches(test_data_batches, test_label_batches, "test")
        test_dataset = [test_data_batches, test_label_batches]

        return test_dataset

    def load_classes(self):

        classes_path = os.path.join(self.root, "classes.txt")

        with open(classes_path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f.readlines()]

        return classes

    def check_spectrogram_batches(self, data_batches, label_batches, split):

        normalization_path = os.path.join(self.root, "spectrogram_normalization.pt")

        if not os.path.exists(normalization_path):
            raise RuntimeError(
                "spectrogram_normalization.pt not found. "
                "Regenerate spectrograms with Dataset/make_garden_spectrogram_pt_dataset.py"
            )

        if len(data_batches) != len(label_batches):
            raise RuntimeError(split + " data and label batches do not match.")

        input_dim = tuple(data_batches[0].shape[1:])

        if input_dim != EXPECTED_INPUT_DIM:
            raise RuntimeError(
                "Expected spectrogram shape "
                + str(EXPECTED_INPUT_DIM)
                + ", got "
                + str(input_dim)
                + ". Regenerate Dataset/mygardenbird_spectrogram_pt."
            )
