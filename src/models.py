import torch
from torch import nn
import math
# Implement a transformer for the task of text classification

class TextClassificationTransformer(nn.Module):
    def __init__(self, w_emsize, d_model, nhead, num_encoder_layers, num_classes, dropout=0.1):
        """
        Arguments:
        :param emsize:
        :param d_hid:
        :param n_class:
        :param nlayers:
        :param dropout:
        """
        super().__init__()
        # First projection layer
        self.proj_layer = nn.Linear(w_emsize, d_model)

        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=1000)

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model, nhead=8)

        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.d_hid = d_model
        self.n_class = num_classes
        self.nlayers = num_encoder_layers
        self.nhead = nhead
        self.softmax = nn.Softmax(dim=1)
        print('Using {} device to train the model.'.format(self.device))

    def add_positional_encoding(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        return  self.positional_encoding(x)


    def forward(self, x, mask=None):
        print(x.dtype)
        x = self.proj_layer(x)
        x = self.add_positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)



