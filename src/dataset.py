# Import torch dataset
from torch.utils.data import Dataset
import os
from utils import preprocess_text
import torch

class LargeMovieDataset(Dataset):
    def __init__(self, path, set='train', embedding_dic=None):
        self.path = path + '/' + set
        self.sentances = []
        self.labels = []
        for label in ['pos', 'neg']:
            for file in os.listdir(self.path + '/' + label):
                with open(self.path + '/' + label + '/' + file, 'r', encoding='utf-8') as f:
                    self.sentances.append(preprocess_text(f.read()))
                self.labels.append(1 if label == 'pos' else 0)
        self.max_len = max([len(sentance) for sentance in self.sentances])
        self.embedding_dic = embedding_dic

        self.data = []
        self.mask = []
        for sentance in self.sentances:
            sentance_data = []
            sentance_mask = []
            for word in sentance:
                sentance_data.append(self.embedding_dic[word])
                sentance_mask.append(1)
            while len(sentance_data) < self.max_len:
                sentance_data.append([0] * 128)
                sentance_mask.append(0)
            self.data.append(sentance_data)
            self.mask.append(sentance_mask)
        self.data = torch.tensor(self.data)
        self.mask = torch.tensor(self.mask)
        self.labels = torch.tensor(self.labels)


    def __len__(self):
        return len(self.sentances)

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx], self.labels[idx]
