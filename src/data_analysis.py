import matplotlib.pyplot as plt
from dataset import LargeMovieDataset
from gensim.models import Word2Vec
from utils import get_sentences_data, check_dir, word_cloud
import numpy as np

if __name__ == '__main__':
    MODELS_DIR = 'models'
    check_dir(MODELS_DIR)
    # Create dataset
    sentences = get_sentences_data(path='../data/aclImdb_v1/aclImdb/train', max_len=10000)
    # Plot length of sentences
    plt.hist([len(sentence) for sentence in sentences], bins=100)
    plt.show()
    word_embedding_size = 16
    sentences = get_sentences_data(path='../data/aclImdb_v1/aclImdb/train')
    word2vec_model = Word2Vec(sentences, vector_size=word_embedding_size, window=3, min_count=1, workers=4)
    word2vec_model.save(f'{MODELS_DIR}/word2vec_model.model')
    print(word2vec_model.wv['any'])
    word2vec_model = Word2Vec.load(f'{MODELS_DIR}/word2vec_model.model')

    train_set = LargeMovieDataset(path='../data/aclImdb_v1/aclImdb', set='train', embedding_dic=word2vec_model.wv, word_embedding_size=word_embedding_size)
    test_set = LargeMovieDataset(path='../data/aclImdb_v1/aclImdb', set='test', embedding_dic=word2vec_model.wv, word_embedding_size=word_embedding_size)
    check_dir('figures')

    # Count number of positive and negative labels

    positive = (np.array(train_set.labels) == 1).sum()
    negative = (np.array(train_set.labels) == 0).sum()
    plt.bar(['positive', 'negative'], [positive, negative])
    plt.savefig('figures/label_distribution.png')
    plt.show()

    positive_sentance = train_set.sentences[train_set.labels == 1]
    negative_sentance = train_set.sentences[train_set.labels == 0]
    word_cloud(positive_sentance, 'positive_word_cloud')
    word_cloud(negative_sentance, 'negative_word_cloud')

