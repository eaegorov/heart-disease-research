from torch.utils.data import DataLoader, Dataset
import pickle
import torch
import torch.nn as nn


class data_loader(Dataset):
    def __init__(self, x_data, y_data):
        with open(x_data, 'rb') as f:
            data = pickle.load(f)

        with open(y_data, 'rb') as f:
            labels = pickle.load(f)

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index, :]
        label = self.labels[index]

        return sample, label
