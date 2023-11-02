import torch.nn as nn
import torch

class SentimentCNNLSTM(nn.Module):
    def __init__(self, word_embedding_size, num_filters, hidden_size, num_layer):
        super(SentimentCNNLSTM, self).__init__()
        self.num_filters = num_filters
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.conv1 = nn.Conv1d(in_channels=word_embedding_size, out_channels=num_filters, kernel_size=3, padding='same')
        self.activation1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.lstm= nn.LSTM(num_filters, hidden_size, num_layer, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
        self.activation2 = nn.Sigmoid()

    def forward(self, x, mask):
        """
        Forward call that include mask
        :param x: batch_size x seq_len x word_embedding_size
        :param mask: batch_size x seq_len
        :return: batch_size x 2
        """
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        # Apply the mask
        mask = self.pool1(mask)
        mask = mask.unsqueeze(2)

        x = x * mask
        # Take the last hidden state of the lstm using the mask by taking the last non zero element

        seq_length = mask.sum(dim=1)
        out = torch.zeros(x.shape[0], 1,x.shape[2])
        for i in range(x.shape[0]):
            out[i] = x[i, seq_length[i].long() - 1, :]
        out = out.squeeze()
        x = self.fc(out)
        x = self.activation2(x)
        return x

    def get_str(self):
        return f'n_filters_{self.num_filters}_h_dim_{self.hidden_size}_n_layer_{self.num_layer}'