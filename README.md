# DeepLearning-UTokyo-2026


## Objective

This projects trains and evaluates a model for bird species classification from their birdsong.

The audio is converted into normalized log-mel spectrogram, then fed into ResNet inspired architecture.



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
## Quick note
This project was made to be run on .ipynb, but as requested on UTOL, we provided the according .py files corresponing. 

## Training

After creating the dataset, either run the `GardenBird_Training.ipynb`  or if you wish to run the python files, 
```bash
pip install -r requirements.txt
python GardenBird_Training.py

```

## Evaluation
Again, either run `GardenBird_Evaluation.ipynb` or run `GardenBird_Evaluation.py`. 


