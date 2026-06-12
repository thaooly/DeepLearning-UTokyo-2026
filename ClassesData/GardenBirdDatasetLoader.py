import csv
import os
import random

import torch
import torchaudio


N_SAMPLES = 48000


class GardenBirdDatasetLoader:
    def __init__(self, root, batch_size=32):

        self.root = root
        self.batch_size = batch_size
        self.classes = []
        self.class_to_idx = {}

    def load_audio_labels_data(self):

        rows = self.load_rows()

        train_dataset = self.load_split_data(rows, "train", shuffle=True)
        val_dataset = self.load_split_data(rows, "val", shuffle=False)

        input_dim = (N_SAMPLES,)
        n_classes = len(self.classes)

        return train_dataset, val_dataset, input_dim, n_classes

    def load_test_audio_labels_data(self):

        rows = self.load_rows()
        return self.load_split_data(rows, "test", shuffle=False)

    def load_split_data(self, rows, split, shuffle):

        rows_split = [row for row in rows if row["split"] == split]

        if len(rows_split) == 0:
            raise RuntimeError("No " + split + " data found in " + self.root)

        if shuffle:
            random.shuffle(rows_split)

        x_batches = []
        y_batches = []

        for start in range(0, len(rows_split), self.batch_size):
            rows_batch = rows_split[start:start + self.batch_size]
            x_batch, y_batch = self.rows_to_batch(rows_batch)
            x_batches.append(x_batch)
            y_batches.append(y_batch)

        return [x_batches, y_batches]

    def rows_to_batch(self, rows):

        x_batch = []
        y_batch = []

        for row in rows:
            waveform = self.load_ogg(row["filepath"])
            label = self.class_to_idx[row["species"]]

            x_batch.append(waveform)
            y_batch.append(label)

        x_batch = torch.stack(x_batch).float()
        y_batch = torch.tensor(y_batch, dtype=torch.long)

        return x_batch, y_batch

    def load_ogg(self, filepath):

        waveform, _ = torchaudio.load(filepath)
        return waveform.squeeze(0).float()

    def load_rows(self):

        metadata_path = os.path.join(self.root, "metadata_subset.csv")
        rows = []

        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                split = row["split"]
                species = row["species"]
                filepath = os.path.join(self.root, row["filepath"])

                rows.append(
                    {
                        "filepath": filepath,
                        "species": species,
                        "split": split,
                    }
                )

        self.classes = sorted(
            list(set([row["species"] for row in rows])), key=lambda x: x.lower()
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        return rows
