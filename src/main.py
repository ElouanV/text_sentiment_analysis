from dataset import LargeMovieDataset
from gensim.models import Word2Vec
from utils import get_sentences_data, check_dir
from torch.utils.data import DataLoader

if __name__ == '__main__':
    MODELS_DIR = 'models'
    check_dir(MODELS_DIR)
    # Create dataset
    word_embedding_size = 16
    word2vec_model = Word2Vec.load(f'{MODELS_DIR}/word2vec_model.model')

    train_set = LargeMovieDataset(path='../aclImdb_v1/', set='train', embedding_dic=word2vec_model.wv, word_embedding_size=word_embedding_size)
    test_set = LargeMovieDataset(path='../aclImdb_v1/', set='test', embedding_dic=word2vec_model.wv, word_embedding_size=word_embedding_size)
