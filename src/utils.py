import string
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import torch
from tqdm import tqdm
import os
import json
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random
from gensim.models import Word2Vec
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
from sklearn.metrics import classification_report

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
        print('Creating directory: ', path)
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


def test(criterion, dataloader, model):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    tot_loss = 0.0
    tot_acc = []
    model.to(device)
    y_pred_tot = []
    y_true_tot = []
    color = "yellow"
    with tqdm(dataloader, unit="batch", colour=color) as tepoch:
        for batch in tepoch:
            data = batch["data"]
            label = batch["label"].squeeze()
            mask = batch["mask"]
            # Put all tensors to device
            data = data.to(device)
            label = label.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                out = model(data, mask)
                # Compute loss
            loss = criterion(out, label)
            # Logging
            tot_loss += loss.item()
            acc = (out.argmax(dim=1) == label.argmax(dim=1)).tolist()
            tot_acc.extend(acc)
            y_pred_tot.extend(out.argmax(dim=1).tolist())
            y_true_tot.extend(label.argmax(dim=1).tolist())

            # Compute precision, recall, f1-score
            acc_str = f"{100. * np.array(tot_acc).mean():.2f}%"
            loss_str = f"{100 * np.array(tot_loss) / len(tot_acc):.2f}"
            tepoch.set_postfix(loss=loss_str, acc=acc_str, set="train")
    stat = classification_report(y_true_tot, y_pred_tot, output_dict=True, target_names=['neg', 'pos'])
    return tot_loss, tot_acc, criterion, dataloader, model, stat


def train_val(run_type, criterion, dataloader, model, optimizer, device, epoch):
    tot_loss = 0.0
    tot_acc = []

    color = "green" if run_type == "train" else "blue"
    with tqdm(dataloader, unit="batch", colour=color) as tepoch:
        for batch in tepoch:
            data = batch["data"]
            label = batch["label"].squeeze()
            mask = batch["mask"]
            # Put all tensors to device
            data = data.to(device)
            label = label.to(device)
            mask = mask.to(device)

            if run_type == "train":
                optimizer.zero_grad()
                out = model(data, mask)
            elif run_type == "val":
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
            acc = (out.argmax(dim=1) == label.argmax(dim=1)).tolist()
            tot_acc.extend(acc)
            if run_type == "test":
                # Compute precision, recall, f1-score
                from sklearn.metrics import classification_report
                print(classification_report(label.argmax(dim=1).tolist(), out.argmax(dim=1).tolist()))
            acc_str = f"{100. * np.array(tot_acc).mean():.2f}%"
            loss_str = f"{100 * np.array(tot_loss) / len(tot_acc):.2f}"
            tepoch.set_postfix(loss=loss_str, acc=acc_str, epoch=epoch, set=run_type)
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
        print(
            f'Train: loss: {epoch_loss / len(train_dataloader):.2f}, accuracy: {np.array(epoch_acc).mean() * 100:.2f}%')
        # Validation
        val_loss, val_acc, criterion, val_dataloader, model, optimizer = train_val(
            "val", criterion, val_dataloader, model, optimizer, device, epoch
        )
        if (np.array(val_acc).mean() > best_eval_acc):
            best_eval_acc = np.array(val_acc).mean()
            torch.save(model.state_dict(), os.path.join('checkpoints', model.__class__.__name__ + '.pth'))

        print(
            f"Validation: loss: {val_loss / len(val_dataloader):.2f}, accuracy: {np.array(val_acc).mean() * 100:.2f}%")
        writer.add_scalar('Validation Loss', val_loss / len(val_dataloader), epoch)
        writer.add_scalar('Validation Accuracy', np.array(val_acc).mean(), epoch)
    writer.close()


def get_model_param(model):
    """
    Get the number of parameters of a models
    :param model: models to get the number of parameters
    :return: int: number of parameters
    """
    return sum(
        param.numel() for param in model.parameters()
    )


def prepare_word2vec(path, save_path, word_embedding_size=128):
    sentences = get_sentences_data(path=os.path.join(path, 'train'), max_len=10000)
    sentences.extend(get_sentences_data(path=os.path.join(path, 'test'), max_len=10000))
    word2vec_model = Word2Vec(sentences, vector_size=word_embedding_size, window=3, min_count=1, workers=4)
    word2vec_model.save(save_path)
    return word2vec_model

def compare_model():
    LOGS_FOLDER = 'logs'
    check_dir(LOGS_FOLDER)

    dfs = []
    for log in os.listdir(LOGS_FOLDER):
        if log.endswith(".json"):
            file_path = os.path.join(LOGS_FOLDER, log)

            with open(file_path, 'r') as file:
                data = json.load(file)

                data['date'] = log.split('_')[2].split('.')[0]
                type = ['neg', 'pos', 'macro avg', 'weighted avg']
                score = ['precision', 'recall', 'f1-score', 'support']
                for t in type:
                    for s in score:
                        data[t.replace(' ', '_') + '_' + s] = data['stat'][t][s]

                del data['stat']

            df = pd.DataFrame([data])

            dfs.append(df)

    if len(dfs) == 0:
        print('[compare_model] No logs found')
        return
    
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv(os.path.join(LOGS_FOLDER, 'logs.csv'), index=False)

def load_model_compare():
    LOGS_FOLDER = 'logs'

    df = pd.read_csv(os.path.join(LOGS_FOLDER, 'logs.csv'))
    return df