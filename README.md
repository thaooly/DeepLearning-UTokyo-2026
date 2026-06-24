# DeepLearning-UTokyo-2026

## Dataset
The data is a collection of bird songs. It is from this paper: https://arxiv.org/abs/2606.06975
To download and build the GardenBird spectrogram dataset, run:

```bash
python Dataset/make_garden_spectrogram_pt_dataset.py
```

This creates:

```text
Dataset/mygardenbird_spectrogram_pt/
```

If the folder already exists, run:

```bash
python Dataset/make_garden_spectrogram_pt_dataset.py --force
```
