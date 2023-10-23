from dataset import LargeMovieDataset
from gensim.models import Word2Vec
from utils import get_sentences_data, check_dir
from torch.utils.data import DataLoader

if __name__ == '__main__':
    MODELS_DIR = 'models'
    check_dir(MODELS_DIR)
    # Create dataset
    sentences = get_sentences_data(path='data/aclImdb_v1/aclImdb/train')
    word_embedding_size = 16
    word2vec_model = Word2Vec(sentences, vector_size=word_embedding_size, window=3, min_count=1, workers=4)
    word2vec_model.save(f'{MODELS_DIR}/word2vec_model.model')
    print(word2vec_model.wv['any'])
    word2vec_model = Word2Vec.load(f'{MODELS_DIR}/word2vec_model.model')

    train_set = LargeMovieDataset(path='data/aclImdb_v1/aclImdb', set='train', embedding_dic=word2vec_model.wv, word_embedding_size=word_embedding_size)
    test_set = LargeMovieDataset(path='data/aclImdb_v1/aclImdb', set='test', embedding_dic=word2vec_model.wv, word_embedding_size=word_embedding_size)

    # Create dataloader

