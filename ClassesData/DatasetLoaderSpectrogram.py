import csv
import os
import warnings

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


SAMPLE_RATE = 16000
N_SAMPLES = 3 * 16000
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 64


class GardenBirdDataset2D(Dataset):
    def __init__(self, rows, class_to_idx):

        super(GardenBirdDataset2D, self).__init__()

        self.rows = rows
        self.class_to_idx = class_to_idx
        self.sample_rate = SAMPLE_RATE
        self.n_fft = N_FFT
        self.hop_length = HOP_LENGTH
        self.n_mels = N_MELS
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):

        row = self.rows[idx]
        waveform = self.load_ogg(row["filepath"])
        spectrogram = self.waveform_to_spectrogram(waveform)
        label = self.class_to_idx[row["species"]]

        return spectrogram, torch.tensor(label, dtype=torch.long)

    def load_ogg(self, filepath):

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="In 2.9, this function's implementation.*"
            )
            waveform, sample_rate = torchaudio.load(filepath)

        if sample_rate != self.sample_rate:
            raise RuntimeError("Expected 16000 Hz audio: " + filepath)

        if waveform.shape[0] != 1:
            raise RuntimeError("Expected mono audio: " + filepath)

        waveform = waveform.squeeze(0)

        if waveform.shape[0] != N_SAMPLES:
            raise RuntimeError("Expected 3 seconds audio: " + filepath)

        return waveform.float()

    def waveform_to_spectrogram(self, waveform):

        waveform = waveform - waveform.mean()
        waveform = waveform / (waveform.std() + 1e-8)

        waveform = waveform.unsqueeze(0)
        spectrogram = self.mel_spectrogram(waveform)
        spectrogram = self.amplitude_to_db(spectrogram)

        spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)

        return spectrogram


class DatasetLoader2D:
    def __init__(self, root, batch_size=32):

        self.root = root
        self.batch_size = batch_size
        self.classes = []
        self.class_to_idx = {}

    def load_spectrogram_labels_data(self):

        train_loader, val_loader, input_dim, n_classes = self.load_torch_dataloaders()

        train_dataset = self.loader_to_batches(train_loader)
        val_dataset = self.loader_to_batches(val_loader)

        return train_dataset, val_dataset, input_dim, n_classes

    def load_test_spectrogram_labels_data(self):

        rows = self.load_rows()
        rows_test = self.rows_for_split(rows, "test")

        dataset_test = GardenBirdDataset2D(rows=rows_test, class_to_idx=self.class_to_idx)

        loader_test = DataLoader(
            dataset_test, batch_size=self.batch_size, shuffle=False
        )

        return self.loader_to_batches(loader_test)

    def load_torch_dataloaders(self):

        rows = self.load_rows()
        rows_train = self.rows_for_split(rows, "train")
        rows_val = self.rows_for_split(rows, "val")

        dataset_train = GardenBirdDataset2D(
            rows=rows_train, class_to_idx=self.class_to_idx
        )

        dataset_val = GardenBirdDataset2D(rows=rows_val, class_to_idx=self.class_to_idx)

        train_loader = DataLoader(
            dataset_train, batch_size=self.batch_size, shuffle=True
        )

        val_loader = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False)

        input_dim = dataset_train[0][0].shape
        n_classes = len(self.classes)

        return train_loader, val_loader, input_dim, n_classes

    def loader_to_batches(self, loader):

        x_batches = []
        y_batches = []

        for x, y in loader:
            x_batches.append(x.float())
            y_batches.append(y.long())

        return [x_batches, y_batches]

    def load_rows(self):

        metadata_path = os.path.join(self.root, "metadata_subset.csv")

        if not os.path.exists(metadata_path):
            raise RuntimeError("metadata_subset.csv not found in " + self.root)

        rows = self.load_rows_from_metadata(metadata_path)

        classes = sorted(
            list(set([row["species"] for row in rows])), key=lambda x: x.lower()
        )
        self.classes = classes
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}

        return rows

    def load_rows_from_metadata(self, metadata_path):

        rows = []

        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                split = row["split"]
                species = row["species"]
                filename = row["filename"]
                filepath = self.find_audio_file(split, species, filename)

                rows.append(
                    {
                        "filepath": filepath,
                        "filename": filename,
                        "species": species,
                        "split": split,
                    }
                )

        return rows

    def rows_for_split(self, rows, split):

        rows_split = [row for row in rows if row["split"] == split]

        if len(rows_split) == 0:
            raise RuntimeError("No " + split + " data found in " + self.root)

        return rows_split

    def find_audio_file(self, split, species, filename):

        filepath = os.path.join(
            self.root, split, self.safe_species_dirname(species), filename
        )

        if os.path.exists(filepath):
            return filepath

        raise RuntimeError("Audio file not found: " + filepath)

    def safe_species_dirname(self, name):

        return name.replace("/", "_").replace("\\", "_").strip()
