import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SentimentTransformer(nn.Transformer):
    def __init__(self, n_input: int, n_output: int, d_model: int, nhead: int, num_encoder_layers: int,
                 dim_feedforward: int):
        super(SentimentTransformer, self).__init__(d_model=d_model, nhead=nhead,
                                               num_encoder_layers=num_encoder_layers,
                                               num_decoder_layers=num_encoder_layers,
                                               dim_feedforward=dim_feedforward)

        self.emb = nn.Embedding(n_input, d_model)
        self.pos_encoder = PositionalEncoding(n_input)
        self.linear = nn.Linear(d_model, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.linear(x)
        return F.sigmoid(x)