from utils import get_sentences_data
import matplotlib.pyplot as plt

sentences = get_sentences_data(path='../data/aclImdb_v1/aclImdb/train', max_len=10000)
# Plot length of sentences
plt.hist([len(sentence) for sentence in sentences], bins=100)
plt.show()

