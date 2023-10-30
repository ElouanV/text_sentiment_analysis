from dataset import LargeMovieDataset
from gensim.models import Word2Vec
from utils import get_sentences_data, check_dir, train, seed_everything, prepare_word2vec
from torch.utils.data import DataLoader
import torch
from models import TextClassificationTransformer

seeds = [3, 7, 42, 666]
if __name__ == '__main__':
    BATCH_SIZE = 32
    train_path = 'data/aclImdb_v1/aclImdb/train'
    data_path = 'data/aclImdb_v1/aclImdb'
    seed_everything(42)
    MODELS_DIR = 'models'
    check_dir(MODELS_DIR)
    # Create dataset
    word_embedding_size = 16
    prepare_word2vec(path=data_path, models_dir=MODELS_DIR, word_embedding_size=128)

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

    # Train model
    model = TextClassificationTransformer(w_emsize=word_embedding_size, nhead=2, d_model=128, num_encoder_layers=3,
                                          dropout=0.1, num_classes=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    train(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs=100)
