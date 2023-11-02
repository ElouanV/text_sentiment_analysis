import os.path

from dataset import LargeMovieDataset
from gensim.models import Word2Vec
from utils import check_dir, train, seed_everything, prepare_word2vec, test
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from model.cnn import TextCNN
import hydra
from omegaconf import OmegaConf
from model.model_loader import get_models
import numpy as np
from datetime import datetime
import json


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))
    BATCH_SIZE = config.batch_size
    word_embedding_sizes = config.word_embedding_sizes
    seeds = config.seeds
    learning_rates = config.learning_rates
    n_epoch = config.n_epoch
    model_name = config.models.name
    model_params = config.models.param
    run_id = config.run_id
    # Print run id in green
    print(f'\033[92mRun id: {run_id}\033[0m')
    print(f'Batch size: {BATCH_SIZE}')
    print(f'Word embedding sizes: {word_embedding_sizes}')
    print(f'Seeds: {seeds}')
    print(f'Learning rates: {learning_rates}')
    print(f'Number of epochs: {n_epoch}')
    print(f'Model name: {model_name}')
    print(f'Model parameters: {model_params}')

    models_dir = 'checkpoints'
    data_path = 'data/aclImdb_v1/aclImdb'
    check_dir(models_dir)
    check_dir('logs')
    check_dir('runs')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}\n')
    for seed in seeds:
        seed_everything(seed)
        for word_embedding_size in word_embedding_sizes:
            word2vec_path = f'{models_dir}/word2vec_model_{word_embedding_size}.model'
            if not os.path.exists(word2vec_path):
                prepare_word2vec(path=data_path, save_path=word2vec_path, word_embedding_size=word_embedding_size)
            word2vec_model = Word2Vec.load(word2vec_path)

            print(f'Loading dataset with word embedding size: {word_embedding_size}')
            train_set = LargeMovieDataset(path=data_path, set='train', embedding_dic=word2vec_model.wv,
                                          word_embedding_size=word_embedding_size)
            test_set = LargeMovieDataset(path=data_path, set='test', embedding_dic=word2vec_model.wv,
                                         word_embedding_size=word_embedding_size)
            print('Creating dataloaders')
            # Split train_set to train and validation with a fixed seed, independent from model training seeds
            train_set, val_set = torch.utils.data.random_split(train_set, [int(0.8 * len(train_set)),
                                                                           len(train_set) - int(0.8 * len(train_set))],
                                                               generator=torch.Generator().manual_seed(42))
            train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
            val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
            test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

            models = get_models(model_name=config.models.name, config=config, word_embedding_size=word_embedding_size,
                                num_classes=2)
            print('Training models...')
            for model in models:
                for learning_rate in learning_rates:
                    criterion = torch.nn.BCELoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    train(model, train_dataloader, val_dataloader, criterion=criterion, optimizer=optimizer,
                          num_epochs=n_epoch)
                    nb_parameters = sum(p.numel() for p in model.parameters())
                    # Test the mode by calling train_eval with test_dataloader and "test" mode
                    tot_loss, tot_acc, criterion, dataloader, model, stat = test(criterion, test_dataloader, model)
                    test_loss = tot_loss / len(dataloader)
                    test_acc = np.array(tot_acc).mean()

                    model_parameters_names = model.get_str()
                    # Save the model
                    model_file_name = f'{config.models.name}_lr_{learning_rate}_s_{seed}_we_size_{word_embedding_size}_{model_parameters_names}.pth'
                    torch.save(model.state_dict(), os.path.join(models_dir, model_file_name))
                    print(f'Model saved to {model_file_name}')
                    print(f'Number of parameters: {nb_parameters}')
                    print(f'Test loss: {test_loss}')
                    print(f'Test accuracy: {test_acc}')

                    model_dict = {
                        'run_id': run_id,
                        'model_name': model_name,
                        'lr': learning_rate,
                        'seed': seed,
                        'word_embedding_size': word_embedding_size,
                        'nb_parameters': nb_parameters,
                        'test_loss': test_loss,
                        'test_acc': test_acc,
                        'model_file_name': model_file_name,
                        'model_parameters_names': model_parameters_names,
                        'stat': stat,
                    }
                    with open(os.path.join('logs', f'model_dict_{datetime.now()}.json'), 'w+') as f:
                        f.write(json.dumps(model_dict, indent=4))
                        f.write('\n')


if __name__ == '__main__':
    import sys

    main()
