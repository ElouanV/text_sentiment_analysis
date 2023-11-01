import os.path

from dataset import LargeMovieDataset
from gensim.models import Word2Vec
from utils import check_dir, train, seed_everything, prepare_word2vec
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from model.cnn import TextCNN
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
    # prepare_word2vec(path=data_path, models_dir=MODELS_DIR, word_embedding_size=word_embedding_size)

    word2vec_model = Word2Vec.load(f'{MODELS_DIR}/word2vec_model.model')

    train_set = LargeMovieDataset(path=data_path, set='train', embedding_dic=word2vec_model.wv,
                                  word_embedding_size=word_embedding_size)
    test_set = LargeMovieDataset(path=data_path, set='test', embedding_dic=word2vec_model.wv,
                                 word_embedding_size=word_embedding_size)

    # Split train_set to train and validation with a fixed seed
    train_set, val_set = torch.utils.data.random_split(train_set, [int(0.8 * len(train_set)),
                                                                   len(train_set) - int(0.8 * len(train_set))], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    # Train models
    model = TextCNN(len_word=word_embedding_size, hidden_size=128, num_classes=2)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train(model, train_dataloader, val_dataloader, criterion=criterion, optimizer=optimizer, num_epochs=10)

    # Load models
    model.load_state_dict(torch.load('checkpoints/SentimentCNN.pth'))

    # Test models on the test set and compute the accuracy
    test_accuracy = 0
    for batch in test_dataloader:
        inputs = batch['data']
        mask = batch['mask']
        labels = batch['label']

        with torch.no_grad():
            predictions = model(inputs, mask)
            _, predictions = torch.max(predictions, 1)
            test_accuracy += torch.sum(predictions == labels)
    test_accuracy = test_accuracy / float(len(test_set))
    print('Test accuracy:', test_accuracy.item())

    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')


if __name__ == '__main__':
    import sys
    main()




