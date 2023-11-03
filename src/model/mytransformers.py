import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, hidden_dim, num_classes, dropout):
        super(TransformerModel, self).__init__()
        self.proj_layer = nn.Linear(input_size, hidden_dim)

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len=1000)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                                        dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Classification Layer
        self.classification_layer = nn.Linear(hidden_dim, num_classes)
        print(f'Nb parameters: {sum(p.numel() for p in self.parameters())}')

    def forward(self, x, mask=None):
        mask = mask
        # x: [batch_size, sequence_length, word_embedding_size]
        x = self.proj_layer(x)
        x = self.pos_encoder(x)
        encoder_output = self.transformer_encoder(x, mask)
        logits = self.classification_layer(encoder_output[:, 0, :])  # Using the [CLS] token for classification
        return F.sigmoid(logits)
