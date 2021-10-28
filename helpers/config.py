# Configuration file for storing settings used across the program.

import tensorflow as tf
tf.random.set_seed(2137)


# model hyperparameters
MAX_WORDS = 998                 # vocabulary size
MAX_LEN = 70                    # length of the post in words
MAX_EPOCHS = 4                 # maximum training time. might terminate sooner.
BATCH_SIZE = 64
LAYER_SIZE = 300
LEARNING_RATE = 0.001
DROPOUT = 0.3
CLASS_WEIGHTS = {0: 1.05, 1: 1}  # class weight for a slightly imbalanced dataset

PRINT_CONVERSATION = True       # print discussion for each test sentence
PRINT_OUTCOME = True            # print outcome of discussion for each test sentence
OPTIMAL_DISCUSSION = True       # restrict discussion to optimal path
PLOTS = True                    # show plots


# sound notification to be played at the end of the program run
def sound_signal():
    import winsound
    duration = 1000  # milliseconds
    freq = 432  # Hz
    winsound.Beep(freq, duration)
