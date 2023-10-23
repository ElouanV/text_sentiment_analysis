# Import torch dataset
from torch.utils.data import Dataset
import os
from utils import preprocess_text
import torch


class LargeMovieDataset(Dataset):
    def __init__(self, path, set='train', embedding_dic=None, max_len=1000, word_embedding_size=128):
        self.path = path + '/' + set
        self.sentences = []
        self.labels = []
        for label in ['pos', 'neg']:
            for file in os.listdir(self.path + '/' + label):
                with open(self.path + '/' + label + '/' + file, 'r', encoding='utf-8') as f:
                    sentence = preprocess_text(f.read())
                    if len(sentence) <= max_len:
                        self.sentences.append(sentence)
                        self.labels.append(1 if label == 'pos' else 0)
        self.max_len = max([len(sentence) for sentence in self.sentences])
        self.embedding_dic = embedding_dic
        self.word_embedding_size = word_embedding_size
        print(f'max_len: {self.max_len}')

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        sentence_data = []
        sentence_mask = []
        for word in sentence:
            sentence_data.append(self.embedding_dic[word])
            sentence_mask.append(1)
        while len(sentence_data) < self.max_len:
            sentence_data.append([0] * self.word_embedding_size)
            sentence_mask.append(0)

        data = torch.tensor(sentence_data)
        mask = torch.tensor(sentence_mask)
        labels = torch.tensor(label)
        return {"data": data, "mask": mask, "labels": labels}

