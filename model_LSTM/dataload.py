import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

class StockDataset(Dataset):
    def __init__(self, data, sequence_length):
        """
        Custom Dataset for stock data.

        Parameters:
            data (pd.DataFrame): DataFrame containing stock features.
            sequence_length (int): Length of the input sequence.
        """
        self.data = data
        self.sequence_length = sequence_length
        self.data = self.data.select_dtypes(include=[np.number]).dropna()
        self.data_values = self.data.values

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        x = self.data_values[index:index + self.sequence_length]
        y = self.data_values[index + self.sequence_length, 0]  # Predict the next Close price
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)