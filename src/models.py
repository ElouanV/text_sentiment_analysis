import torch
from torch import nn

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
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.d_hid = d_model
        self.n_class = num_classes
        self.nlayers = num_encoder_layers
        self.nhead = nhead
        self.embedding = nn.Embedding(w_emsize, d_model)
        self.softmax = nn.Softmax(dim=1)
        print('Using {} device to train the model.'.format(self.device))


    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Reshape for Transformer
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Average across tokens
        x = self.fc(x)
        x = self.softmax(x)
        return x


