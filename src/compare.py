import random
import time
import torch
import torch.nn as nn

from gensim.models import Word2Vec
from dataset import LargeMovieDataset
from torch.utils.data import DataLoader
from utils import train, seed_everything

#from rnn import Model

def generate_seed(num_seeds: int) -> list[int]:
    seed_list = []
    for _ in range(num_seeds):
        seed = random.randint(0, 2**32 - 1)
        seed_list.append(seed)
    return seed_list

def execution_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, elapsed_time

# A retirer j'ai pas reussi a import depuis le notebook rnn 
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer):
        super(Model, self).__init__()
        self.input_size = input_size        
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.c1 = nn.RNN(self.input_size, self.hidden_size, num_layers=num_layer, batch_first=True)
        self.c2 = nn.Linear(self.hidden_size, 2)

    def forward(self, x, mask):
        x, hiden = self.c1(x)
        x = x[:, -1, :]
        x = self.c2(x)
        return F.relu(x)
#########

def data_loader_rrn(MODEL_PATH, DATA_PATH):
    word_embedding_size = 16
    word2vec_model = Word2Vec.load(MODEL_PATH + "word2vec_model.models")

    train_set = LargeMovieDataset(path=DATA_PATH, set="train", embedding_dic=word2vec_model.wv, word_embedding_size=word_embedding_size)
    test_set = LargeMovieDataset(path=DATA_PATH, set='test', embedding_dic=word2vec_model.wv, word_embedding_size=word_embedding_size)

    train_set, val_set = torch.utils.data.random_split(train_set, [int(0.8 * len(train_set)), len(train_set) - int(0.8 * len(train_set))])

    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=32, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader

def model_config_rnn():
    word_embedding_size = 16
    hyp_hidensize = 128
    num_layer = 1
    rnn_model = Model(word_embedding_size, hyp_hidensize, num_layer)

    learnong_rate=1e-2
    optimizer_rnn = torch.optim.Adam(rnn_model.parameters(), lr=learnong_rate)
    criterion_rnn = nn.CrossEntropyLoss()

    return rnn_model, optimizer_rnn, criterion_rnn

def data_loader_cnn():
    pass

def modele_config_cnn():
    pass

def train_model(model_list, nb):
    MODEL_PATH = 'model/'
    DATA_PATH = './data/aclImdb_v1/aclImdb'

    seed = generate_seed(nb)

    model_dict = {}

    if 'rnn_model' in model_list:
        train_dataloader, val_dataloader, test_dataloader = data_loader_rrn(MODEL_PATH, DATA_PATH)
        rnn_model, optimizer_rnn, criterion_rnn = model_config_rnn()
        #train(models=rnn_model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, optimizer=optimizer_rnn, criterion=criterion_rnn, num_epochs=2)

        times = []
        results = []
        for i in range(nb):
            seed_everything(seed[i])
            result, time = execution_time(train, rnn_model, train_dataloader, val_dataloader, optimizer_rnn, criterion_rnn, 2)
            times.append(time)
            results.append(result)
        
        model_dict['rnn_model'] = {'time': times, 'result': results}
    if 'ccn_model' in model_list:
        pass

    return model_dict

def compare_model():
    pass

def main():
    train_model(['rnn_model'])    

if __name__ == '__main__':
    main()