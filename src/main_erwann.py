from dataset import LargeMovieDataset
from cnn import SentimentCNN
from sklearn.feature_extraction.text import TfidfVectorizer
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
    word_embedding_size = 16
    word2vec_model = Word2Vec.load(f'{MODELS_DIR}/word2vec_model.model')

    train_set = LargeMovieDataset(path='../aclImdb_v1/', set='train', embedding_dic=word2vec_model.wv, word_embedding_size=word_embedding_size)
    test_set = LargeMovieDataset(path='../aclImdb_v1/', set='test', embedding_dic=word2vec_model.wv, word_embedding_size=word_embedding_size)

    # split the training set into training and validation set (80 for training, 20 for validation)
    train_set, validation_set = torch.utils.data.random_split(train_set, [int(len(train_set) * 0.8), len(train_set) - int(len(train_set) * 0.8)])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

    print("======================================")
    print("Data loaded")
    print('Training set size:', len(train_set))
    print('Validation set size:', len(validation_set))
    print('Test set size:', len(test_set))
    print("======================================")

    cnnModel = SentimentCNN(vocab_size=10000, embedding_dim=word_embedding_size, hidden_dim=100)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnnModel.parameters(), lr=0.001)

    train(cnnModel, train_loader, validation_loader, criterion, optimizer, num_epochs=10)

    # test accuracy with the test set
    test_accuracy = 0
    for inputs, labels in test_loader:
        predictions = cnnModel(inputs)
        _, predictions = torch.max(predictions, 2)
        test_accuracy += torch.sum(predictions == labels) / labels.size()[0]
    test_accuracy /= len(test_loader)
    print('Test accuracy:', test_accuracy.item())
