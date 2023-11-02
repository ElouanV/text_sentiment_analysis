import os
import subprocess

if __name__ == '__main__':
    models= ['TextCNN', 'LSTM', 'RNN', 'SeqCNN', 'CNNLSTM']
    processess = []
    for model in models:
        cmd = f'python src/main_grid_search.py models={model} > exec_model_{model}.log'
        print(cmd)
        # Execute cmd in another thread
        processess.append(subprocess.Popen(cmd, shell=True))

    # Wait for all threads to finish
    for process in processess:
        process.wait()
    print('Finished all processes')


import torch
print(torch.cuda.is_available()
)
print(f'torch version: {torch.__version__}')