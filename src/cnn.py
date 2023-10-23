import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class SentimentCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SentimentCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, input_seq):
        embeds = self.embedding(input_seq)
        output, _ = self.lstm(embeds)
        output = self.output(output)
        return output

    def evaluate(self, inputs, labels):
        predictions = self.forward(inputs)
        _, predictions = torch.max(predictions, 2)
        self.accuracy = torch.sum(predictions == labels) / labels.size()[0]

def trainingModelCNN(model, training_dataloader, valid_dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        logging_loss = 0.0
        for i, data in enumerate(training_dataloader):
            input = data['data']
            labels = data['label']

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            out = model(input)

            # Compute loss
            loss = criterion(out, labels)

            # Compute gradients
            loss.backward()

            # Backward pass - model update
            optimizer.step()

            logging_loss += loss.item()

            if i % 2000 == 1999:
                # Logging training loss
                logging_loss /= 2000
                print('Training loss epoch ', epoch, ' -- mini-batch ', i, ': ', logging_loss)
                logging_loss = 0.0
            
                # Model validation
                with torch.no_grad():
                    logging_loss_val = 0.0
                    for data_val in tqdm(valid_dataloader):
                        input_val, labels_val = data_val
                        out_val = model(input_val)
                        loss_val = criterion(out_val, labels_val)
                        logging_loss_val += loss_val.item()
                    logging_loss_val /= len(valid_dataloader)
                    print('Validation loss: ', logging_loss_val)