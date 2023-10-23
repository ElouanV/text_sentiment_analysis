from utils import get_sentances_data
import matplotlib.pyplot as plt

sentances = get_sentances_data(path='../data/aclImdb_v1/aclImdb/train', max_len=10000)
# Plot length of sentances
plt.hist([len(sentance) for sentance in sentances], bins=100)
plt.show()