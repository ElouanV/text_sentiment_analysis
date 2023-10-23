# Import torch dataset
from torch.utils.data import Dataset
import os
class LargeMovieDataset(Dataset):
    def __init__(self, path, set='train'):
        self.path = path + '/' + set
        self.data = []
        self.labels = []

        for label in ['pos', 'neg']:
            for file in os.listdir(self.path + '/' + label):
                with open(self.path + '/' + label + '/' + file, 'r') as f:
                    self.data.append(f.read())
                    self.labels.append(1 if label == 'pos' else 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]