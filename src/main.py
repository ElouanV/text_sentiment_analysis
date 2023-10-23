from dataset import LargeMovieDataset
from gensim.models import Word2Vec
from utils import get_sentances_data, check_dir


if __name__ == '__main__':
    MODELS_DIR = 'models'
    check_dir(MODELS_DIR)
    # Create dataset
    sentances = get_sentances_data(path='data/aclImdb_v1/aclImdb/train')
    word_embedding_size = 16
    word2vec_model = Word2Vec(sentances, vector_size=word_embedding_size, window=3, min_count=1, workers=4)
    word2vec_model.save(f'{MODELS_DIR}/word2vec_model.model')
    word2vec_model = Word2Vec.load(f'{MODELS_DIR}/word2vec_model.model')

    dataset = LargeMovieDataset(path='data/aclImdb_v1/aclImdb', set='train', embedding_dic=word2vec_model.wv, word_embedding_size=word_embedding_size)
    print('Dataset size:', len(dataset))
    print('First sample:', dataset[0])
    print('Second sample:', dataset[1])
    print('Last sample:', dataset[-1])
    print('Second to last sample:', dataset[-2])