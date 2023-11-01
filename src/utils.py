import string
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import torch
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random
from gensim.models import Word2Vec
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

nltk.download('punkt')


def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()


def get_sentences_data(path, max_len=1000):
    sentences = []
    for label in ['pos', 'neg']:
        for file in os.listdir(path + '/' + label):
            with open(path + '/' + label + '/' + file, 'r', encoding='utf-8') as f:
                sentence = preprocess_text(f.read())
                if len(sentence) <= max_len:
                    sentences.append(sentence)
    return sentences


def word_cloud(train, filename):
    long_string = ','.join(list(train))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue',
                          width=800, height=400)
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    print("Wordcloud for our whole dataset:")
    wordcloud.to_image()
    wordcloud.to_file(f"figures/{filename}.png")
    plt.imshow(wordcloud, interpolation='bilinear')


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


def train_val(run_type, criterion, dataloader, model, optimizer, device, epoch):
    tot_loss = 0.0
    tot_acc = []

    color = "green" if run_type == "train" else "blue"
    with tqdm(dataloader, unit="batch", colour=color) as tepoch:
        for batch in tepoch:
            data = batch["data"]
            label = batch["label"]
            mask = batch["mask"]
            # Put all tensors to device
            data = data.to(device)
            label = label.to(device)
            mask = mask.to(device)

            if run_type == "train":
                optimizer.zero_grad()
                out = model(data, mask)
            elif run_type == "val" or run_type == "test":
                with torch.no_grad():
                    out = model(data, mask)
                # Compute loss
            loss = criterion(out, label)
            if run_type == "train":
                # Compute gradients
                loss.backward()
                # Backward pass - models update
                optimizer.step()

            # Logging
            tot_loss += loss.item()
            acc = (out.argmax(dim=1) == label).tolist()
            tot_acc.extend(acc)
            acc_str = f"{100. * np.array(acc).mean():.2f}%"
            tepoch.set_postfix(loss=loss.item(), acc=acc_str, epoch=epoch, set=run_type)
    return tot_loss, tot_acc, criterion, dataloader, model, optimizer


def train(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs=10):
    """
    Train the models

    Args:
        model: models to train
        train_dataloader: dataloader for training
        val_dataloader: dataloader for validation
        optimizer: optimizer to use
        criterion: loss function
        num_epochs: number of epochs to train

    Returns:
        None
    """
    check_dir('checkpoints')
    check_dir('runs')
    best_eval_acc = 0.0
    # Create a TensorBoard summary writer to log data with a specific filename model_name_datetime
    writer = SummaryWriter(log_dir=f'runs/{model.__class__.__name__}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f'Using device: {device} to train\n')
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}/{num_epochs}")
        # Training
        epoch_loss, epoch_acc, criterion, train_dataloader, model, optimizer = train_val(
            "train", criterion, train_dataloader, model, optimizer, device, epoch
        )
        # print(f"Epoch {epoch}, loss: {epoch_loss / len(train_dataloader)}, accuracy: {np.array(epoch_acc).mean()}%")
        writer.add_scalar('Training Loss', epoch_loss / len(train_dataloader), epoch)
        writer.add_scalar('Training Accuracy', np.array(epoch_acc).mean(), epoch)
        print(f'Train: loss: {epoch_loss / len(train_dataloader)}, accuracy: {np.array(epoch_acc).mean() * 100}%')
        # Validation
        val_loss, val_acc, criterion, val_dataloader, model, optimizer = train_val(
            "val", criterion, val_dataloader, model, optimizer, device, epoch
        )
        if (np.array(val_acc).mean() > best_eval_acc):
            best_eval_acc = np.array(val_acc).mean()
            torch.save(model.state_dict(), os.path.join('checkpoints', model.__class__.__name__ + '.pth'))
        print(f"Validation: loss: {val_loss / len(val_dataloader)}, accuracy: {np.array(val_acc).mean() * 100}%")
        writer.add_scalar('Validation Loss', val_loss / len(val_dataloader), epoch)
        writer.add_scalar('Validation Accuracy', np.array(val_acc).mean(), epoch)
    writer.close()


def test(model, test_dataloader, criterion):
    """
    Test the models

    Args:
        model: models to test
        test_dataloader: dataloader for testing
        criterion: loss function

    Returns:
        None
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f'Using device: {device} to test\n')
    test_loss, test_acc, criterion, test_dataloader, model, optimizer = train_val(
        "test", criterion, test_dataloader, model, None, device, 0
    )
    print(f"Test: {test_loss / len(test_dataloader)}, {np.array(test_acc).mean()}")


def get_model_param(model):
    """
    Get the number of parameters of a models
    :param model: models to get the number of parameters
    :return: int: number of parameters
    """
    return sum(
        param.numel() for param in model.parameters()
    )


def prepare_word2vec(path, models_dir, word_embedding_size=128):
    sentences = get_sentences_data(path=os.path.join(path, 'train'), max_len=10000)
    sentences.extend(get_sentences_data(path=os.path.join(path, 'test'), max_len=10000))
    word2vec_model = Word2Vec(sentences, vector_size=word_embedding_size, window=3, min_count=1, workers=4)
    word2vec_model.save(f'{models_dir}/word2vec_model.model')
    return word2vec_model
