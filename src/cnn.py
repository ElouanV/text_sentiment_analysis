import torch
import torch.nn as nn
import torch.optim as optim

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
    
def train(model, data, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = 0
    
    for epoch in range(epochs):
        for inputs, labels in data:
            model.zero_grad()
            predictions = model(inputs)
            loss = nn.CrossEntropyLoss()(predictions.view(-1, 2), labels)
            loss.backward()
            optimizer.step()
        print('Epoch:', epoch, 'Loss:', loss.item())

    return model