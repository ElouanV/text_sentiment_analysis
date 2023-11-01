# Import torch dataset
from torch.utils.data import Dataset
import os
from utils import preprocess_text
import torch
import numpy as np
import os


class LargeMovieDataset(Dataset):
    def __init__(self, path, set='train', embedding_dic=None, max_len=1000, word_embedding_size=128):
        self.path = os.path.join(path, set)
        assert embedding_dic is not None, 'embedding_dic must be provided'
        self.sentences = []
        self.labels = []
        for label in ['pos', 'neg']:
            for file in os.listdir(os.path.join(self.path, label)):
                with open(os.path.join(self.path, label, file), 'r', encoding='utf-8') as f:
                    sentence = preprocess_text(f.read())
                    if len(sentence) <= max_len:
                        self.sentences.append(sentence)
                        self.labels.append(1 if label == 'pos' else 0)
        self.max_len = max_len
        self.embedding_dic = embedding_dic
        self.word_embedding_size = word_embedding_size
        print(f'Max length of the sentences: {self.max_len}')

    def __len__(self):
        """
        Return the length of the dataset
        Returns: int: length of the dataset
        """
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        Return the data and label of the idx-th sample
        :param idx: int: index of the sample
        Returns: dict: {"data": data, "mask": mask, "label": label}
        """
        sentence = self.sentences[idx]
        label = self.labels[idx]
        sentence_data = np.zeros((self.max_len, self.word_embedding_size))
        sentence_mask = np.zeros(self.max_len)
        for i, word in enumerate(sentence):
            sentence_data[i] = self.embedding_dic[word] if word in self.embedding_dic else np.zeros(self.word_embedding_size)
            sentence_mask[i] = 1
        
        sentence_data = np.array(sentence_data)
        sentence_mask = np.array(sentence_mask)

        data = torch.tensor(sentence_data, dtype=torch.float)
        mask = torch.tensor(sentence_mask, dtype=torch.float)
        label = torch.tensor(label)
        return {"data": data, "mask": mask, "label": label}

