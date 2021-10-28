from helpers import config as cf
from helpers.data_loader import get_data, introduce_variation
from model_lstm import LSTMDebater
from model_bdir import BIDIRDebater
from helpers.protocol import WHO

mx_len = [30, 60, 100]
epochs = [12]
batch = [32, 64, 100]
mx_words = [1000]
main_lr_size = [50, 70, 90, 120]
lr = [0.0005, 0.001, 0.005]
dropout = [0.3, 0.5]

import itertools
a = [mx_words, dropout, mx_len, epochs, batch, main_lr_size, lr]
combinations = list(itertools.product(*a))
# combinations = combinations[:3]

import csv

# open the file in the write mode
with open('overnight_results.csv', 'w', newline='') as f:
    # create the csv writer
    writer = csv.writer(f)
    header = ["mx_words", "dropout", "mx_len", "epochs", "batch", "main_lr_size", "lr",
              "bestLSTm", "bestBDIR"]
    writer.writerow(header)

    counter = 0
    for param_set in combinations:
        cf.MAX_WORDS = param_set[0]
        cf.DROPOUT = param_set[1]
        cf.MAX_LEN = param_set[2]
        cf.MAX_EPOCHS = param_set[3]
        cf.BATCH_SIZE = param_set[4]
        cf.LAYER_SIZE = param_set[5]
        cf.LEARNING_RATE = param_set[6]

        data_1 = get_data()
        data_2 = get_data()
        data_2 = introduce_variation(data_2)

        lstm = LSTMDebater(WHO.Model1, data_1)
        bdir = BIDIRDebater(WHO.Model2, data_2)
        bestLSTM = lstm.run()
        bestBDIR = bdir.run()
        newlist = list(param_set)
        newlist.append(bestLSTM)
        newlist.append(bestBDIR)
        writer.writerow(newlist)
        f.flush()
        print(counter)
        counter += 1

