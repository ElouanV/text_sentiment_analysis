#!/bin/bash

# Define the list of models
models=('TextCNN' 'SeqCNN' 'RNN' 'LSTM')

# Loop over the models and call the Python script
for model in "${models[@]}"; do
    python src/main_grid_search.py models="$model"
done
