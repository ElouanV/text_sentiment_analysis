# Deep Learning Class Project: Text Sentiment Analysis Benchmark
## Introduction

Welcome to the Deep Learning class project on benchmarking various deep learning methods for text sentiment analysis using the Large Movie Review Dataset. In this project, we will explore and compare the performance of different neural network architectures, including TextCNN, SeqCNN, CNNLSTM, RNN, BiLSTM, LSTM, and Transformer encoder, for the task of sentiment analysis on movie reviews.

The goal of this project is to provide an in-depth evaluation of these architectures on the sentiment analysis task, enabling you to understand their strengths, weaknesses, and trade-offs. This README will guide you through the project setup, data, and usage.
## Dataset

The dataset used in this project is the Large Movie Review Dataset, commonly known as the IMDB dataset. It contains 25,000 movie reviews for training and 25,000 for testing, with a binary sentiment classification task (positive or negative sentiment). You can obtain the dataset from the following URL: [Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

The dataset is pre-processed and divided into train and test sets. In this project we focused on the supervised version of the dataset.
## Project Structure

The project is structured as follows:

    dataset.py: contains the Dataset class
    models/: This directory will contain Python files for each of the deep learning architectures (TextCNN, SeqCNN, CNNLSTM, RNN, BiLSTM, LSTM, Transformer encoder).
    main_grid_search.py: The main training script. You can specify the model architecture and training parameters as command-line arguments.
    conf/: contains models and training configuration
    requirements.txt: A list of Python packages required to run the project.

## Getting Started

    Clone this repository to your local machine.

    Create a virtual environment and install the required packages using pip:

    
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

Download the dataset and place it in the data/ directory.

Modify the model architecture and hyperparameters in the corresponding model files within the conf/ directory

Train a model using the main_grid_search.py script by specifying the architecture as a command-line argument:

```bash

python src/main_grid_search.py models=TextCNN
```
After training, you can either use tensorboard:
```bash
python -m tensorboard.main --logdir=runs/
```
Or parse `logs/` using function provided in `utils/`

## Dataset citation
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
