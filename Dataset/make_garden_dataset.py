#!/usr/bin/env python3
"""
Download MyGardenBird 16 kHz from Zenodo and create a balanced OGG subset.

Default output:
    mygardenbird_ogg_12x200/
      train/<species>/*.ogg
      val/<species>/*.ogg
      test/<species>/*.ogg
      metadata_subset.csv

Default subset:
    12 species × 200 clips/species = 2400 clips
    split: 80% train, 10% val, 10% test

Requirements:
    pip install requests pandas tqdm

You also need ffmpeg:
    Ubuntu/Debian:
        sudo apt update && sudo apt install ffmpeg

    Conda:
        conda install -c conda-forge ffmpeg
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm


DEFAULT_RECORD_ID = "20306877"
DEFAULT_OUT_DIR = "mygardenbird_ogg_12x200"
DEFAULT_CACHE_DIR = "mygardenbird_download_cache"
DEFAULT_N_PER_CLASS = 200
DEFAULT_SEED = 42
DEFAULT_MAX_MB = 100.0
DEFAULT_OGG_QUALITY = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MyGardenBird 16 kHz and build a balanced OGG subset."
    )
    parser.add_argument("--record-id", default=DEFAULT_RECORD_ID)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--n-per-class", type=int, default=DEFAULT_N_PER_CLASS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-mb", type=float, default=DEFAULT_MAX_MB)
    parser.add_argument(
        "--ogg-quality",
        type=int,
        default=DEFAULT_OGG_QUALITY,
        help=(
            "Vorbis quality from -1 to 10. "
            "2 or 3 is usually a good compromise. Higher = better/larger."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate the output folder if it already exists.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep the downloaded zip and extracted full dataset after creating the subset.",
    )
    parser.add_argument(
        "--auto-reduce",
        action="store_true",
        help=(
            "If the final folder is above --max-mb, delete train clips evenly "
            "across species until the folder is below the limit."
        ),
    )
    return parser.parse_args()


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not installed or not in PATH.\n"
            "Install it with one of:\n"
            "  sudo apt update && sudo apt install ffmpeg\n"
            "  conda install -c conda-forge ffmpeg"
        )


def get_zenodo_files(record_id: str) -> list[dict]:
    url = f"https://zenodo.org/api/records/{record_id}"
    print(f"[zenodo] Reading record metadata: {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()["files"]


def choose_16khz_zip(files: Iterable[dict]) -> dict:
    candidates = []
    for f in files:
        name = f.get("key", "").lower()
        if name.endswith(".zip") and any(x in name for x in ["16k", "16khz", "16_khz"]):
            candidates.append(f)

    if not candidates:
        print("\nCould not auto-detect the 16 kHz zip. Files available:")
        for f in files:
            print(" -", f.get("key"))
        raise RuntimeError("No 16 kHz zip found in the Zenodo record.")

    candidates.sort(key=lambda x: x.get("size", 0), reverse=True)
    return candidates[0]


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] Already downloaded: {dest}")
        return

    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with (
            open(dest, "wb") as f,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
            ) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extracted"

    if marker.exists():
        print(f"[skip] Already extracted: {extract_dir}")
        return

    print(f"[extract] {zip_path} -> {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    marker.write_text("ok\n", encoding="utf-8")


def find_audio_root(extract_dir: Path) -> Path:
    """
    Expected structure after extraction:
        extracted_16k/mygardenbird16khz/Asian Koel/*.wav
        extracted_16k/mygardenbird16khz/Collared Kingfisher/*.wav
        ...
    """
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

    if best_root is None or best_count == 0:
        raise RuntimeError(f"No audio folders with WAV files found under {extract_dir}")

    return best_root


def folder_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def safe_species_dirname(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").strip()


def split_counts_for(n_per_class: int) -> dict[str, int]:
    n_train = int(round(n_per_class * 0.80))
    n_val = int(round(n_per_class * 0.10))
    n_test = n_per_class - n_train - n_val

    return {
        "train": n_train,
        "val": n_val,
        "test": n_test,
    }


def convert_wav_to_ogg(src: Path, dst: Path, quality: int) -> None:
    """
    Keep 16 kHz mono, convert WAV PCM to OGG Vorbis.
    -vn: no video
    -ar 16000: force 16 kHz
    -ac 1: mono
    -q:a quality: Vorbis VBR quality
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "libvorbis",
        "-q:a",
        str(quality),
        str(dst),
    ]

    subprocess.run(cmd, check=True)


def build_ogg_subset(
    audio_root: Path,
    out_dir: Path,
    n_per_class: int,
    seed: int,
    ogg_quality: int,
    force: bool,
) -> pd.DataFrame:
    if out_dir.exists():
        if not force:
            raise RuntimeError(
                f"Output folder already exists: {out_dir}\n"
                f"Run again with --force if you want to overwrite it."
            )
        print(f"[delete] Existing output folder: {out_dir}")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    species_dirs = sorted(
        [p for p in audio_root.iterdir() if p.is_dir() and any(p.glob("*.wav"))],
        key=lambda p: p.name.lower(),
    )

    if not species_dirs:
        raise RuntimeError(f"No species folders found in {audio_root}")

    print(f"\n[species found] {len(species_dirs)}")
    for d in species_dirs:
        print(f" - {d.name}: {len(list(d.glob('*.wav')))} wav files")

    counts = split_counts_for(n_per_class)

    print("\n[split per species]")
    for split, n in counts.items():
        print(f" {split}: {n}")

    rows = []

    for species_dir in species_dirs:
        species = species_dir.name
        species_safe = safe_species_dirname(species)
        wavs = sorted(species_dir.glob("*.wav"), key=lambda p: p.name)

        if len(wavs) < n_per_class:
            raise RuntimeError(
                f"Not enough WAV files for species '{species}': "
                f"{len(wavs)} found, {n_per_class} needed."
            )

        rng.shuffle(wavs)
        selected = wavs[:n_per_class]

        start = 0
        for split, n_split in counts.items():
            split_files = selected[start : start + n_split]
            start += n_split

            dst_dir = out_dir / split / species_safe
            dst_dir.mkdir(parents=True, exist_ok=True)

            for src in tqdm(split_files, desc=f"{species} / {split}", unit="clip"):
                dst = dst_dir / (src.stem + ".ogg")
                convert_wav_to_ogg(src, dst, quality=ogg_quality)

                rows.append(
                    {
                        "filepath": str(dst),
                        "filename": dst.name,
                        "species": species,
                        "split": split,
                        "source_wav": str(src),
                        "ogg_quality": ogg_quality,
                    }
                )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(out_dir / "metadata_subset.csv", index=False)
    return metadata


def auto_reduce_to_limit(
    out_dir: Path, metadata: pd.DataFrame, max_mb: float
) -> pd.DataFrame:
    """
    Delete train clips evenly across species until folder size <= max_mb.
    Keeps val/test unchanged.
    """
    max_bytes = max_mb * 1_000_000
    current_size = folder_size_bytes(out_dir)

    if current_size <= max_bytes:
        return metadata

    print(
        f"\n[auto-reduce] Current size is {current_size / 1_000_000:.1f} MB, "
        f"above {max_mb:.1f} MB."
    )
    print("[auto-reduce] Removing train clips evenly across species...")

    metadata = metadata.copy()

    while folder_size_bytes(out_dir) > max_bytes:
        train_df = metadata[metadata["split"] == "train"].copy()
        if train_df.empty:
            raise RuntimeError("Cannot reduce further: no train clips left.")

        counts = train_df.groupby("species").size().sort_values(ascending=False)
        species_to_reduce = counts.index[0]

        candidates = train_df[train_df["species"] == species_to_reduce]
        row_idx = candidates.index[-1]
        file_path = Path(metadata.loc[row_idx, "filepath"])

        if file_path.exists():
            file_path.unlink()

        metadata = metadata.drop(index=row_idx)

    metadata = metadata.reset_index(drop=True)
    metadata.to_csv(out_dir / "metadata_subset.csv", index=False)

    print(f"[auto-reduce] New size: {folder_size_bytes(out_dir) / 1_000_000:.1f} MB")
    return metadata


def print_summary(out_dir: Path, metadata: pd.DataFrame, max_mb: float) -> None:
    total_bytes = folder_size_bytes(out_dir)
    total_mb = total_bytes / 1_000_000
    total_mib = total_bytes / (1024 * 1024)

    print("\n[done]")
    print(f"Output folder: {out_dir}")
    print(f"Metadata:      {out_dir / 'metadata_subset.csv'}")
    print(f"Total clips:   {len(metadata)}")
    print(f"Size:          {total_mb:.1f} MB / {total_mib:.1f} MiB")

    print("\n[counts]")
    counts = metadata.groupby(["species", "split"]).size().unstack(fill_value=0)
    print(counts)

    if total_mb > max_mb:
        print(
            f"\n[warning] The subset is {total_mb:.1f} MB, "
            f"which is above your limit of {max_mb:.1f} MB."
        )
        print("Try for example:")
        print(
            "  python make_mygardenbird_ogg_subset.py --n-per-class 180 --ogg-quality 2 --force"
        )
        print("or add:")
        print("  --auto-reduce")
    else:
        print(f"\n[ok] Under the configured limit: {max_mb:.1f} MB")


def main() -> None:
    args = parse_args()

    if args.ogg_quality < -1 or args.ogg_quality > 10:
        raise RuntimeError("--ogg-quality must be between -1 and 10.")

    require_ffmpeg()

    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        files = get_zenodo_files(args.record_id)
        zip_info = choose_16khz_zip(files)

        zip_name = zip_info["key"]
        zip_size_mb = zip_info.get("size", 0) / 1_000_000
        zip_url = zip_info["links"]["self"]
        zip_path = cache_dir / zip_name

        print(f"[selected] {zip_name} ({zip_size_mb:.1f} MB)")

        download_file(zip_url, zip_path)

        extract_dir = cache_dir / "extracted_16k"
        extract_zip(zip_path, extract_dir)

        audio_root = find_audio_root(extract_dir)
        print(f"[audio_root] {audio_root}")

        metadata = build_ogg_subset(
            audio_root=audio_root,
            out_dir=out_dir,
            n_per_class=args.n_per_class,
            seed=args.seed,
            ogg_quality=args.ogg_quality,
            force=args.force,
        )

        if args.auto_reduce:
            metadata = auto_reduce_to_limit(out_dir, metadata, max_mb=args.max_mb)

        print_summary(out_dir, metadata, max_mb=args.max_mb)

        if not args.keep_cache:
            print(f"\n[cleanup] Removing cache folder: {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)
        else:
            print(f"\n[cache kept] {cache_dir}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
