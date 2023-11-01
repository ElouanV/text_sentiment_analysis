from dataset import LargeMovieDataset
from model.cnn import TextCNN, SequentialCNN
from dataset import LargeMovieDataset
from gensim.models import Word2Vec
from utils import get_sentences_data, check_dir, train
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

if __name__ == '__main__':
    MODELS_DIR = 'models'
    check_dir(MODELS_DIR)
    # Create dataset

    print("Word2Vec")

    sentences = get_sentences_data(path='../aclImdb_v1/train')
    word_embedding_size = 16
    word2vec_model = Word2Vec(sentences, vector_size=word_embedding_size, window=3, min_count=1, workers=4)
    word2vec_model.save(f'{MODELS_DIR}/word2vec_model.model')

    word2vec_model = Word2Vec.load(f'{MODELS_DIR}/word2vec_model.model')

    print("Loading data")

    train_set = LargeMovieDataset(path='../aclImdb_v1/', set='train', embedding_dic=word2vec_model.wv, word_embedding_size=word_embedding_size)
    test_set = LargeMovieDataset(path='../aclImdb_v1/', set='test', embedding_dic=word2vec_model.wv, word_embedding_size=word_embedding_size)

    # split the training set into training and validation set (80 for training, 20 for validation)
    train_set, validation_set = torch.utils.data.random_split(train_set, [int(len(train_set) * 0.8), len(train_set) - int(len(train_set) * 0.8)])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

    cnnModel = SequentialCNN(len_word=word_embedding_size, hidden_size=32, num_classes=2, dropout=0.5)
    optimizer = optim.Adam(cnnModel.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Training")

    train(cnnModel, train_loader, validation_loader, optimizer, criterion, num_epochs=10)
