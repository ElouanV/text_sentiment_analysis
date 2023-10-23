import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class SentimentCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, n_filters, filter_sizes, pool_size, hidden_size, num_classes,dropout):
        super().__init__()        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embed_size)) for fs in filter_sizes])
        self.max_pool1 = nn.MaxPool1d(pool_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(95*n_filters, hidden_size, bias=True)  
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=True)  

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)       
        convolution = [conv(embedded) for conv in self.convs]   
        max1 = self.max_pool1(convolution[0].squeeze()) 
        max2 = self.max_pool1(convolution[1].squeeze())
        cat = torch.cat((max1, max2), dim=2)      
        x = cat.view(cat.shape[0], -1) 
        x = self.fc1(self.relu(x))
        x = self.dropout(x)
        x = self.fc2(x)  
        return x

def trainingModelCNN(model, training_dataloader, valid_dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        logging_loss = 0.0
        for i, data in enumerate(training_dataloader):
            input = data['data']
            labels = data['label']
            mask = data['mask']


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