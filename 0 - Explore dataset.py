import os
import torch
from ClassesData.DatasetLoader import DatasetLoader
from Utilities.Utilities import Utilities

# FASHION MNIST #
path_parent_project = os.getcwd()
dataset_image_path = path_parent_project + "/Dataset/" + "FASHION/"

dataset = DatasetLoader(root=dataset_image_path)
train_dataset, val_dataset, input_dim, n_classes = dataset.load_images_labels_data()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
images = train_dataset[0][0].to(device)
labels = train_dataset[1][0].to(device)

Utilities.images_as_canvas(images=images)

# TEXT #
dataset = DatasetLoader(root="")
dataset, tokenizer = dataset.load_text_data()

vocab = tokenizer.get_vocab()
sorted_vocab = sorted(vocab.items(), key=lambda x: x[0])

word_to_tokenize = "example cat"
tokens = tokenizer.encode(word_to_tokenize)

for token_id in tokens:
    token = tokenizer.convert_ids_to_tokens(token_id)
    print("Token ID: " + str(token_id) + " - Token: " + token)

sample = 0
text = dataset["train"]["text"][sample]
label = dataset["train"]["label"][sample]

print(text)
print(label)
