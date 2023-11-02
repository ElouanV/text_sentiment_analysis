import os.path

from dataset import LargeMovieDataset
from gensim.models import Word2Vec
from utils import check_dir, train, seed_everything, prepare_word2vec, test
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from model.cnnlstm import SentimentCNNLSTM
import hydra
from omegaconf import OmegaConf

seeds = [3, 7, 42, 666]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))
    BATCH_SIZE = 32
    train_path = 'data/aclImdb_v1/aclImdb/train'
    data_path = 'data/aclImdb_v1/aclImdb'
    seed_everything(3)
    MODELS_DIR = 'checkpoints'
    check_dir(MODELS_DIR)
    # Create dataset
    word_embedding_size = 64

    word2vec_path = f'{MODELS_DIR}/word2vec_model_{word_embedding_size}.model'
    if not os.path.exists(word2vec_path):
        prepare_word2vec(path=data_path, save_path=word2vec_path, word_embedding_size=word_embedding_size)
    word2vec_model = Word2Vec.load(f'{MODELS_DIR}/word2vec_model.model')

    train_set = LargeMovieDataset(path=data_path, set='train', embedding_dic=word2vec_model.wv,
                                  word_embedding_size=word_embedding_size)
    test_set = LargeMovieDataset(path=data_path, set='test', embedding_dic=word2vec_model.wv,
                                 word_embedding_size=word_embedding_size)

    # Split train_set to train and validation with a fixed seed
    train_set, val_set = torch.utils.data.random_split(train_set, [int(0.8 * len(train_set)),
                                                                   len(train_set) - int(0.8 * len(train_set))],
                                                       generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    # Train models
    model = SentimentCNNLSTM(word_embedding_size=word_embedding_size, num_filters=64, hidden_size=64, num_layer=1)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # train(model, train_dataloader, val_dataloader, criterion=criterion, optimizer=optimizer, num_epochs=10)

    # Load models
    model.load_state_dict(torch.load('checkpoints/SentimentCNNLSTM.pth'))

    # Test models on the test set and compute the accuracy
    tot_loss, tot_acc, criterion, dataloader, model, stat = test(criterion, test_dataloader, model, device=torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"))
    print(stat)


if __name__ == '__main__':
    import sys

    main()
