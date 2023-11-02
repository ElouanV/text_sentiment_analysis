import torch.nn as nn
import torch.nn.functional as F
import torch

class SentimentRNN(nn.Module):
    """Hyperparameters : [hiden_size, num_layer]"""
    def __init__(self, input_size, hidden_size, num_layer):
        super(SentimentRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.rnn = nn.RNN(self.input_size, self.hidden_size, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 2)

    def forward(self, x, mask):
        x, hiden = self.rnn(x)
        #USE THE MASK 
        x = [x[batch_id, int(sum(mask[batch_id])) -1, :] for batch_id in range(len(x))]
        x = torch.stack((*x,), dim=0)
        x = self.fc(x)
        return F.sigmoid(x)
    def get_str(self):
        return f'h_dim_{self.hidden_size}_n_l_{self.num_layer}'


class SentimentLSTM(nn.Module):
    """Hyperparameters : [hiden_size, num_layer, bidirectional]"""
    def __init__(self, input_size, hidden_size, num_layer, bidirectional_bool):
        super(SentimentLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.bidirectional_bool = bidirectional_bool
        self.lstm = nn.LSTM(self.input_size, int(self.hidden_size / (bidirectional_bool + 1)), num_layers=num_layer, batch_first=True, bidirectional=bidirectional_bool)
        self.fc = nn.Linear(self.hidden_size, 2)

    def forward(self, x, mask):
        x, hiden = self.lstm(x)
        #USE THE MASK 
        x = [x[batch_id, int(sum(mask[batch_id])) -1, :] for batch_id in range(len(x))]
        x = torch.stack((*x,), dim=0)
        x = self.fc(x)
        return F.sigmoid(x)

    def get_str(self):
        return f'h_dim_{self.hidden_size}_n_l_{self.num_layer}_bi_{int(self.bidirectional_bool)}'