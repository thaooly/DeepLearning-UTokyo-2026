import os

# Set environment variables
os.environ['HF_DATASETS_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from transformers import AutoTokenizer

class DatasetLoader:

    def __init__(self, root):

        self.root = root

    def load_images_labels_data(self):

        # Load data tensors based on image size
        train_data_batches = torch.load(os.path.join(self.root, 'train_data_batches.pt'))
        train_label_batches = torch.load(os.path.join(self.root, 'train_label_batches.pt'))

        val_data_batches = torch.load(os.path.join(self.root, 'val_data_batches.pt'))
        val_label_batches = torch.load(os.path.join(self.root, 'val_label_batches.pt'))

        train_data_batches = [batch.float() for batch in train_data_batches]
        val_data_batches = [batch.float() for batch in val_data_batches]

        train_dataset = [train_data_batches, train_label_batches]
        val_dataset = [val_data_batches, val_label_batches]

        input_dim = train_dataset[0][0].numpy().shape[1:]
        n_classes = 10

        return train_dataset, val_dataset, input_dim, n_classes

    def load_text_data(self):

        # Load dataset
        dataset = load_dataset('ag_news')

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        return dataset, tokenizer

    def load_HAR_data(self):

        data = torch.load(os.path.join(self.root, 'data_tensor.pt'))
        labels = torch.load(os.path.join(self.root, 'labels_tensor.pt'))

        indices = [i for i in range(len(labels))]

        np.random.shuffle(indices)  # Shuffle the indices randomly
        split_index = int(0.6 * len(indices))  # Calculate the split index
        train_indices = indices[:split_index]  # First 80% for training
        test_indices = indices[split_index:]  # Remaining 20% for testing

        data_X_train = data[train_indices][:, :, :, np.newaxis]
        data_Y_train = labels[train_indices]

        data_X_test = data[test_indices][:, :, :, np.newaxis]
        data_Y_test = labels[test_indices]

        input_dim = data_X_train.shape[1:]
        n_classes = 8

        train_dataset = [[data_X_train], [data_Y_train]]
        val_dataset = [[data_X_test], [data_Y_test]]

        return train_dataset, val_dataset, input_dim, n_classes

# Custom dataset class for tokenization
class AGNewsDataset(Dataset):

    def __init__(self, dataset, tokenizer, max_len):

        super(AGNewsDataset, self).__init__()

        self.texts = dataset['text']
        self.labels = dataset['label']
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        texts = str(self.texts[idx])
        labels = self.labels[idx]

        encoding = self.tokenizer(texts,
                                  add_special_tokens=True,
                                  max_length=self.max_len,
                                  truncation=True,
                                  padding='max_length',
                                  return_tensors='pt')

        return {'input_text': texts,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'targets': torch.tensor(labels, dtype=torch.long)}
