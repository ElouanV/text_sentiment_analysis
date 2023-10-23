# Import torch dataset
from torch.utils.data import Dataset
import os
from utils import preprocess_text
import torch


class LargeMovieDataset(Dataset):
    def __init__(self, path, set='train', embedding_dic=None, max_len=1000, word_embedding_size=128):
        self.path = path + '/' + set
        self.sentances = []
        self.labels = []
        for label in ['pos', 'neg']:
            for file in os.listdir(self.path + '/' + label):
                with open(self.path + '/' + label + '/' + file, 'r') as f:
                    sentance = preprocess_text(f.read())
                    if len(sentance) <= max_len:
                        self.sentances.append(sentance)
                        self.labels.append(1 if label == 'pos' else 0)
        self.max_len = max([len(sentance) for sentance in self.sentances])
        self.embedding_dic = embedding_dic
        self.word_embedding_size = word_embedding_size
        print(f'max_len: {self.max_len}')

    def __len__(self):
        return len(self.sentances)

    def __getitem__(self, idx):
        sentance = self.sentances[idx]
        label = self.labels[idx]
        sentance_data = []
        sentance_mask = []
        for word in sentance:
            sentance_data.append(self.embedding_dic[word])
            sentance_mask.append(1)
        while len(sentance_data) < self.max_len:
            sentance_data.append([0] * self.word_embedding_size)
            sentance_mask.append(0)

        data = torch.tensor(sentance_data)
        mask = torch.tensor(sentance_mask)
        labels = torch.tensor(label)
        return data, mask, labels

