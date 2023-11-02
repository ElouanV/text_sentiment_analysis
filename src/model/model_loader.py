from model.cnn import TextCNN, SequentialCNN
from model.rnn import SentimentRNN, SentimentLSTM
from model.mytransformers import SentimentTransformer
from model.cnnlstm import SentimentCNNLSTM

def load_textcnn(config, word_embedding_size, num_classes):
    """

    :param config:
    :param word_embedding_size:
    :param num_classes:
    :return:
    """
    hidden_sizes = config.models.param['hidden_size']
    models = []
    for hidden_size in hidden_sizes:
        model = TextCNN(len_word=word_embedding_size, hidden_size=hidden_size, num_classes=num_classes)
        models.append(model)
    return models


def load_rnn(config, word_embedding_size, num_classes):
    hidden_sizes = config.models.param['hidden_size']
    num_layers = config.models.param['num_layers']
    models = []
    for hidden_size in hidden_sizes:
        for num_layer in num_layers:
            model = SentimentRNN(input_size=word_embedding_size, hidden_size=hidden_size, num_layer=num_layer)
            models.append(model)
    return models


def load_lstm(config, word_embedding_size, num_classes):
    hidden_sizes = config.models.param['hidden_size']
    num_layers = config.models.param['num_layers']
    bidirectional_bools = config.models.param['bidirectional']
    models = []
    for hidden_size in hidden_sizes:
        for num_layer in num_layers:
            for bidirectional_bool in bidirectional_bools:
                model = SentimentLSTM(input_size=word_embedding_size, hidden_size=hidden_size, num_layer=num_layer,
                                      bidirectional_bool=bidirectional_bool)
                models.append(model)
    return models


def load_sequentialcnn(config, word_embedding_size, num_classes):
    hidden_sizes = config.models.param['hidden_size']
    models = []
    for hidden_size in hidden_sizes:
        model = SequentialCNN(len_word=word_embedding_size, hidden_size=hidden_size, num_classes=num_classes)
        models.append(model)
    return models


def load_transformer(config, word_embedding_size, num_classes):
    return []

def load_cnnlstm(config, word_embedding_size, num_classes):
    num_filters = config.models.param['num_filters']
    hidden_sizes = config.models.param['hidden_size']
    num_layers = config.models.param['num_layers']
    models = []
    for hidden_size in hidden_sizes:
        for num_layer in num_layers:
            for num_filter in num_filters:
                model = SentimentCNNLSTM(word_embedding_size=word_embedding_size, num_filters=num_filter, hidden_size=hidden_size, num_layer=num_layer)
                models.append(model)
    return models


def get_models(model_name, config, word_embedding_size, num_classes):
    """
    Get the models according to the name
    :param model_name: name of the models
    :return: models
    """
    if model_name == 'textcnn':
        return load_textcnn(config, word_embedding_size, num_classes)
    if model_name == 'rnn':
        return load_rnn(config, word_embedding_size, num_classes)
    if model_name == 'lstm':
        return load_lstm(config, word_embedding_size, num_classes)
    if model_name == 'seqcnn':
        return load_sequentialcnn(config, word_embedding_size, num_classes)
    if model_name == 'transformer':
        return load_transformer(config, word_embedding_size, num_classes)
    if model_name == 'cnnlstm':
        return load_cnnlstm(config, word_embedding_size, num_classes)
    raise Exception(f'Unknown models name: {model_name}')
