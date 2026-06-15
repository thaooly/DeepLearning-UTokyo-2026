#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


DEFAULT_AUDIO_DIR = "Dataset/mygardenbird_pt"
DEFAULT_OUT_DIR = "Dataset/mygardenbird_spectrogram_pt"
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 160
N_MELS = 128
F_MIN = 150
F_MAX = 8000
TOP_DB = 80.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create spectrogram PT batches from GardenBird audio PT batches."
    )
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_transforms():
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX,
        power=2.0,
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
        stype="power",
        top_db=TOP_DB,
    )

    return mel_spectrogram, amplitude_to_db


def waveform_to_logmel(
    waveform: torch.Tensor,
    mel_spectrogram: torchaudio.transforms.MelSpectrogram,
    amplitude_to_db: torchaudio.transforms.AmplitudeToDB,
) -> torch.Tensor:
    waveform = waveform - waveform.mean()
    waveform = waveform.unsqueeze(0)

    spectrogram = mel_spectrogram(waveform)
    spectrogram = amplitude_to_db(spectrogram)

    return spectrogram


def compute_train_stats(
    audio_dir: Path,
    mel_spectrogram: torchaudio.transforms.MelSpectrogram,
    amplitude_to_db: torchaudio.transforms.AmplitudeToDB,
) -> tuple[torch.Tensor, torch.Tensor]:
    data_batches = torch.load(audio_dir / "train_data_batches.pt")

    total_sum = torch.tensor(0.0)
    total_square_sum = torch.tensor(0.0)
    total_count = 0

    for batch in tqdm(data_batches, desc="train stats"):
        for waveform in batch:
            spectrogram = waveform_to_logmel(
                waveform,
                mel_spectrogram,
                amplitude_to_db,
            )
            total_sum += spectrogram.sum()
            total_square_sum += (spectrogram * spectrogram).sum()
            total_count += spectrogram.numel()

    mean = total_sum / total_count
    variance = total_square_sum / total_count - mean * mean
    std = torch.sqrt(variance.clamp_min(1e-6))

    return mean, std


def convert_split(
    audio_dir: Path,
    out_dir: Path,
    split: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    mel_spectrogram: torchaudio.transforms.MelSpectrogram,
    amplitude_to_db: torchaudio.transforms.AmplitudeToDB,
) -> None:
    data_batches = torch.load(audio_dir / (split + "_data_batches.pt"))
    label_batches = torch.load(audio_dir / (split + "_label_batches.pt"))

    spectrogram_batches = []

    for batch in tqdm(data_batches, desc=split):
        spectrogram_batch = []

        for waveform in batch:
            spectrogram = waveform_to_logmel(
                waveform,
                mel_spectrogram,
                amplitude_to_db,
            )
            spectrogram = (spectrogram - mean) / std
            spectrogram_batch.append(spectrogram)

        spectrogram_batches.append(torch.stack(spectrogram_batch).float())

    torch.save(spectrogram_batches, out_dir / (split + "_data_batches.pt"))
    torch.save(label_batches, out_dir / (split + "_label_batches.pt"))


def main() -> None:
    args = parse_args()

    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)

    if out_dir.exists():
        if not args.force:
            raise RuntimeError("Output folder already exists: " + str(out_dir))
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(audio_dir / "classes.txt", out_dir / "classes.txt")
    shutil.copy2(audio_dir / "metadata_subset.csv", out_dir / "metadata_subset.csv")

    mel_spectrogram, amplitude_to_db = build_transforms()
    mean, std = compute_train_stats(audio_dir, mel_spectrogram, amplitude_to_db)

    torch.save(
        {
            "mean": mean,
            "std": std,
            "sample_rate": SAMPLE_RATE,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "f_min": F_MIN,
            "f_max": F_MAX,
            "top_db": TOP_DB,
        },
        out_dir / "spectrogram_normalization.pt",
    )

    for split in ["train", "val", "test"]:
        convert_split(
            audio_dir,
            out_dir,
            split,
            mean,
            std,
            mel_spectrogram,
            amplitude_to_db,
        )

    print("[done] " + str(out_dir))


if __name__ == "__main__":
    main()
