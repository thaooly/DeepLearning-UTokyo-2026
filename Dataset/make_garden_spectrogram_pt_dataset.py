#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import requests
import torch
import torchaudio
from tqdm import tqdm


DEFAULT_RECORD_ID = "20306877"
DEFAULT_OUT_DIR = "Dataset/mygardenbird_spectrogram_pt"
DEFAULT_CACHE_DIR = "Dataset/mygardenbird_download_cache"
DEFAULT_N_PER_CLASS = 575
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 42
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 160
N_MELS = 128
F_MIN = 150
F_MAX = 8000
TOP_DB = 80.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download MyGardenBird 16 kHz and save balanced log-mel "
            "spectrogram PT batches."
        )
    )
    parser.add_argument("--record-id", default=DEFAULT_RECORD_ID)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--n-per-class", type=int, default=DEFAULT_N_PER_CLASS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-cache", action="store_true")
    return parser.parse_args()


def get_zenodo_files(record_id: str) -> list[dict]:
    url = f"https://zenodo.org/api/records/{record_id}"
    print("[zenodo] " + url)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()["files"]


def choose_16khz_zip(files: Iterable[dict]) -> dict:
    candidates = []

    for f in files:
        name = f.get("key", "").lower()
        if name.endswith(".zip") and any(x in name for x in ["16k", "16khz", "16_khz"]):
            candidates.append(f)

    if len(candidates) == 0:
        raise RuntimeError("No 16 kHz zip found in Zenodo record.")

    candidates.sort(key=lambda x: x.get("size", 0), reverse=True)
    return candidates[0]


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print("[skip] " + str(dest))
        return

    print("[download] " + url)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with open(dest, "wb") as f:
            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extracted"

    if marker.exists():
        print("[skip] " + str(extract_dir))
        return

    print("[extract] " + str(zip_path))
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    marker.write_text("ok\n", encoding="utf-8")


def find_audio_root(extract_dir: Path) -> Path:
    best_root = None
    best_count = 0

    for candidate in extract_dir.rglob("*"):
        if not candidate.is_dir():
            continue

        species_dirs = [
            p for p in candidate.iterdir() if p.is_dir() and any(p.glob("*.wav"))
        ]

        if len(species_dirs) > best_count:
            best_count = len(species_dirs)
            best_root = candidate

    if best_root is None:
        raise RuntimeError("No WAV folders found under " + str(extract_dir))

    return best_root


def split_counts_for(n_per_class: int) -> dict[str, int]:
    n_train = int(round(n_per_class * 0.80))
    n_val = int(round(n_per_class * 0.10))
    n_test = n_per_class - n_train - n_val

    return {
        "train": n_train,
        "val": n_val,
        "test": n_test,
    }


def record_id_for(wav_path: Path) -> str:
    return wav_path.stem.split("_")[0]


def split_wavs_by_record(
    wavs: list[Path],
    counts: dict[str, int],
    rng: random.Random,
) -> dict[str, list[Path]]:
    groups_by_record = {}

    for wav_path in wavs:
        record_id = record_id_for(wav_path)
        groups_by_record.setdefault(record_id, []).append(wav_path)

    record_groups = []

    for record_id in sorted(groups_by_record.keys()):
        record_group = sorted(groups_by_record[record_id], key=lambda p: p.name)
        rng.shuffle(record_group)
        record_groups.append(record_group)

    rng.shuffle(record_groups)

    split_files = {
        "train": [],
        "val": [],
        "test": [],
    }

    for record_group in record_groups:
        deficits = {
            split: counts[split] - len(split_files[split])
            for split in split_files.keys()
        }
        possible_splits = {
            split: deficit
            for split, deficit in deficits.items()
            if 0 < len(record_group) <= deficit
        }

        if len(possible_splits) > 0:
            split = max(possible_splits, key=possible_splits.get)
        else:
            split = max(deficits, key=deficits.get)

        if deficits[split] <= 0:
            break

        split_files[split].extend(record_group)

    return split_files


def load_wav(filepath: Path) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(filepath))

    if sample_rate != SAMPLE_RATE:
        raise RuntimeError(
            "Expected "
            + str(SAMPLE_RATE)
            + " Hz audio, got "
            + str(sample_rate)
            + " Hz for "
            + str(filepath)
        )

    return waveform.squeeze(0).float()


def build_rows(
    audio_root: Path,
    n_per_class: int,
    rng: random.Random,
) -> tuple[list[dict], list[str]]:
    counts = split_counts_for(n_per_class)

    species_dirs = sorted(
        [p for p in audio_root.iterdir() if p.is_dir() and any(p.glob("*.wav"))],
        key=lambda p: p.name.lower(),
    )
    classes = [p.name for p in species_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    rows = []

    for species_dir in species_dirs:
        wavs = sorted(species_dir.glob("*.wav"), key=lambda p: p.name)

        if len(wavs) < n_per_class:
            raise RuntimeError("Not enough files for " + species_dir.name)

        rng.shuffle(wavs)
        split_files = split_wavs_by_record(wavs, counts, rng)

        for split in ["train", "val", "test"]:
            for wav_path in split_files[split]:
                rows.append(
                    {
                        "filename": wav_path.name,
                        "species": species_dir.name,
                        "split": split,
                        "label": class_to_idx[species_dir.name],
                        "source_wav": wav_path,
                    }
                )

    return rows, classes


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
    rows: list[dict],
    mel_spectrogram: torchaudio.transforms.MelSpectrogram,
    amplitude_to_db: torchaudio.transforms.AmplitudeToDB,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_rows = [row for row in rows if row["split"] == "train"]

    total_sum = torch.tensor(0.0)
    total_square_sum = torch.tensor(0.0)
    total_count = 0

    for row in tqdm(train_rows, desc="train stats"):
        waveform = load_wav(row["source_wav"])
        spectrogram = waveform_to_logmel(
            waveform,
            mel_spectrogram,
            amplitude_to_db,
        )
        total_sum += spectrogram.sum()
        total_square_sum += (spectrogram * spectrogram).sum()
        total_count += spectrogram.numel()

    if total_count == 0:
        raise RuntimeError("No train samples found.")

    mean = total_sum / total_count
    variance = total_square_sum / total_count - mean * mean
    std = torch.sqrt(variance.clamp_min(1e-6))

    return mean, std


def save_spectrogram_batches(
    rows: list[dict],
    out_dir: Path,
    split: str,
    batch_size: int,
    rng: random.Random,
    mean: torch.Tensor,
    std: torch.Tensor,
    mel_spectrogram: torchaudio.transforms.MelSpectrogram,
    amplitude_to_db: torchaudio.transforms.AmplitudeToDB,
) -> list[dict]:
    x_batches = []
    y_batches = []

    split_rows = [row for row in rows if row["split"] == split]
    rng.shuffle(split_rows)

    for start in tqdm(range(0, len(split_rows), batch_size), desc=split):
        batch_rows = split_rows[start : start + batch_size]
        x_batch = []
        y_batch = []

        for row in batch_rows:
            waveform = load_wav(row["source_wav"])
            spectrogram = waveform_to_logmel(
                waveform,
                mel_spectrogram,
                amplitude_to_db,
            )
            spectrogram = (spectrogram - mean) / std

            x_batch.append(spectrogram)
            y_batch.append(row["label"])

        x_batches.append(torch.stack(x_batch).float())
        y_batches.append(torch.tensor(y_batch, dtype=torch.long))

    torch.save(x_batches, out_dir / (split + "_data_batches.pt"))
    torch.save(y_batches, out_dir / (split + "_label_batches.pt"))

    return split_rows


def save_metadata(out_dir: Path, classes: list[str], rows: list[dict]) -> None:
    with open(out_dir / "classes.txt", "w", encoding="utf-8") as f:
        for name in classes:
            f.write(name + "\n")

    with open(
        out_dir / "metadata_subset.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        fieldnames = ["filename", "species", "split", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "filename": row["filename"],
                    "species": row["species"],
                    "split": row["split"],
                    "label": row["label"],
                }
            )


def save_normalization(out_dir: Path, mean: torch.Tensor, std: torch.Tensor) -> None:
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


def build_spectrogram_dataset(
    audio_root: Path,
    out_dir: Path,
    n_per_class: int,
    batch_size: int,
    seed: int,
    force: bool,
) -> None:
    if out_dir.exists():
        if not force:
            raise RuntimeError("Output folder already exists: " + str(out_dir))
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    rows, classes = build_rows(audio_root, n_per_class, rng)
    mel_spectrogram, amplitude_to_db = build_transforms()
    mean, std = compute_train_stats(rows, mel_spectrogram, amplitude_to_db)

    save_normalization(out_dir, mean, std)

    saved_rows = []

    for split in ["train", "val", "test"]:
        saved_rows += save_spectrogram_batches(
            rows,
            out_dir,
            split,
            batch_size,
            rng,
            mean,
            std,
            mel_spectrogram,
            amplitude_to_db,
        )

    save_metadata(out_dir, classes, saved_rows)
    print("[done] " + str(out_dir))


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)

    if out_dir.exists() and not args.force:
        raise RuntimeError("Output folder already exists: " + str(out_dir))

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = get_zenodo_files(args.record_id)
    zip_info = choose_16khz_zip(files)
    zip_path = cache_dir / zip_info["key"]

    download_file(zip_info["links"]["self"], zip_path)

    extract_dir = cache_dir / "extracted_16k"
    extract_zip(zip_path, extract_dir)

    audio_root = find_audio_root(extract_dir)
    print("[audio_root] " + str(audio_root))

    build_spectrogram_dataset(
        audio_root=audio_root,
        out_dir=out_dir,
        n_per_class=args.n_per_class,
        batch_size=args.batch_size,
        seed=args.seed,
        force=args.force,
    )

    if not args.keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
