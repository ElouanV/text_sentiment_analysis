import string
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import torch
from tqdm import tqdm
import os

nltk.download('punkt')

def get_sentences_data(path, max_len=1000):
    sentences = []
    for label in ['pos', 'neg']:
        for file in os.listdir(path + '/' + label):
            with open(path + '/' + label + '/' + file, 'r', encoding='utf-8') as f:
                sentence = preprocess_text(f.read())
                if len(sentence) <= max_len:
                    sentences.append(sentence)
    return sentences


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def preprocess_text(sentence):
    # Convert to lowercase
    sentence = sentence.lower()

    # Remove <*> with regex
    import re
    sentence = re.sub('<[^<]*?>', '', sentence)
    # Remove punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    return words


def forward_pass(name, mask, model):
    out = [None] * name.shape[0]
    hidden = torch.zeros(name.shape[0], 57)
    for i in range(name.shape[1]):
        character = name[:, i].squeeze(1)
        out_, hidden = model(character, hidden)

        for batch_id in range(name.shape[0]):
            if mask[batch_id, i] == 1:
                out[batch_id] = out_[batch_id].unsqueeze(0)
    out = torch.cat(out, dim=0)
    return out


def train_val(run_type, criterion, dataloader, model, optimizer):
    tot_loss = 0.0
    tot_acc = []
    for mb_idx, batch in tqdm(enumerate(dataloader)):
        name = batch["data"]
        label = batch["label"]
        mask = batch["mask"]

        if run_type == "train":
            # zero the parameter gradients
            optimizer.zero_grad()

        # Forward pass
        if run_type == "train":
            out = forward_pass(name, mask, model)
        elif run_type == "val":
            with torch.no_grad():
                out = forward_pass(name, mask, model)

        # Compute loss
        loss = criterion(out, label)

        if run_type == "train":
            # Compute gradients
            loss.backward()

            # Backward pass - model update
            optimizer.step()

        # Logging
        tot_loss += loss.item()
        acc = (out.argmax(dim=1) == label).tolist()
        tot_acc.extend(acc)
    return tot_loss, tot_acc, criterion, dataloader, model, optimizer


def train(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs=10):
    """
    Train the model

    Args:
        model: model to train
        train_dataloader: dataloader for training
        val_dataloader: dataloader for validation
        optimizer: optimizer to use
        criterion: loss function
        num_epochs: number of epochs to train

    Returns:
        None
    """
    for epoch in range(num_epochs):
        # Training
        epoch_loss, epoch_acc, criterion, train_dataloader, model, optimizer = train_val(
            "train", criterion, train_dataloader, model, optimizer
        )
        print(
            f"Epoch {epoch}: {epoch_loss/len(train_dataloader)}, {np.array(epoch_acc).mean()}"
        )
    
        # Validation
        val_loss, val_acc, criterion, val_dataloader, model, optimizer = train_val(
            "val", criterion, val_dataloader, model, optimizer
        )
        print(f"Val: {val_loss/len(val_dataloader)}, {np.array(val_acc).mean()}")
