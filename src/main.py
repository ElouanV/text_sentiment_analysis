from dataset import LargeMovieDataset
from gensim.models import Word2Vec
from utils import get_sentences_data, check_dir, train
from torch.utils.data import DataLoader
import torch
from models import TextClassificationTransformer

if __name__ == '__main__':
    MODELS_DIR = 'models'
    check_dir(MODELS_DIR)
    # Create dataset
    sentences = get_sentences_data(path='data/aclImdb_v1/aclImdb/train')
    word_embedding_size = 16
    word2vec_model = Word2Vec(sentences, vector_size=word_embedding_size, window=3, min_count=1, workers=4)
    word2vec_model.save(f'{MODELS_DIR}/word2vec_model.model')

    word2vec_model = Word2Vec.load(f'{MODELS_DIR}/word2vec_model.model')

    train_set = LargeMovieDataset(path='data/aclImdb_v1/aclImdb', set='train', embedding_dic=word2vec_model.wv, word_embedding_size=word_embedding_size)
    test_set = LargeMovieDataset(path='data/aclImdb_v1/aclImdb', set='test', embedding_dic=word2vec_model.wv, word_embedding_size=word_embedding_size)


    # Split train_set to train and validation
    train_set, val_set = torch.utils.data.random_split(train_set, [int(0.8 * len(train_set)),
                                                                   len(train_set) - int(0.8 * len(train_set))])

    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=32, shuffle=True)

    # Train model
    model = TextClassificationTransformer()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    train(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs=100)






