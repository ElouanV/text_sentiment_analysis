import torch.nn as nn
import torch.nn.functional as F

class SentimentRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer):
        super(SentimentRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.c1 = nn.RNN(self.input_size, self.hidden_size, num_layers=num_layer, batch_first=True)
        self.c2 = nn.Linear(self.hidden_size, 2)

    def forward(self, x, mask):
        x, hiden = self.c1(x)
        x = x[:, -1, :]
        x = self.c2(x)
        return F.relu(x)


class SentimentLSTM(nn.Module):
    def __init__(self, bidriectionnal=False):
        super(SentimentLSTM).__init__()
        raise Exception('Not implemented')