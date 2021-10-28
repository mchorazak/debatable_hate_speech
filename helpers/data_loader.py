from Data import Data
import pandas as pd


# Load data and process. Split into sets.
# Return a Data object.
def get_data():
    import pandas as pd

    df = pd.read_csv("data/data_preprocessed_with_stopwords.csv")

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from helpers.config import MAX_LEN, MAX_WORDS, PLOTS

    # tokenize words. word turned into int
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(df["text"])
    sequences = tokenizer.texts_to_sequences(df["text"])
    # pad sentences shorter than MAX_LEN with 0
    tweets = pad_sequences(sequences, maxlen=MAX_LEN)

    # print histogram of sentence length
    if PLOTS:
        import matplotlib.pyplot as plt
        sentence_lengths = []
        for i in range(len(sequences)):
            sentence_lengths.append(len(sequences[i]))
        plt.hist(sentence_lengths, bins=200)
        plt.xlabel('Number of words')
        plt.ylabel('Number of sentences')
        plt.show()

    df["label"].replace({"hate": 1, "nothate": 0}, inplace=True)
    text = pd.DataFrame(tweets)
    new = pd.concat([df, text], axis=1)

    # Create x and y datasets for train, dev(val) and test.
    train_set = new.loc[(df['split'] == "train")]
    train_set = train_set.reset_index(drop=True)
    x_train = train_set.iloc[:, 13:]
    y_train = pd.DataFrame(train_set["label"])

    val_set = new.loc[(df['split'] == "dev")]
    val_set = val_set.reset_index(drop=True)
    x_val = val_set.iloc[:, 13:]
    y_val = pd.DataFrame(val_set["label"])

    test_set = new.loc[df['split'] == "test"]
    test_set = test_set.reset_index(drop=True)
    x_test = test_set.iloc[:, 13:]
    y_test = pd.DataFrame(test_set["label"])

    # save textual representations too for printing to the user
    test_sentences = test_set["text"]
    train_sentences = train_set["text"]
    val_sentences = val_set["text"]

    return Data(x_train, y_train, x_val, y_val, x_test, y_test,
                test_sentences, train_sentences, val_sentences)


# Shuffle training and validation datasets.
def introduce_variation(data):
    # concatenate training data, labels, and original text sentences
    df_train = pd.concat([pd.DataFrame(data.y_train), pd.DataFrame(data.train_sentences),
                          pd.DataFrame(data.x_train)], axis=1)
    df_val = pd.concat([pd.DataFrame(data.y_val), pd.DataFrame(data.val_sentences), pd.DataFrame(data.x_val)], axis=1)

    # calculate ratios
    ratio = len(df_val) / (len(df_train) + len(df_val))

    # append validation set to training set for shuffling
    df = df_train.append(df_val, ignore_index=True)

    # split the val+train into separate datasets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.reset_index(drop=True)
    new_val = df.sample(frac=ratio,  random_state=42)
    new_train = df.drop(new_val.index)

    # split val and train into data, labels and text sentences to print
    data.y_train = new_train.iloc[:, 0:1]
    data.y_val = new_val.iloc[:, 0:1]
    from helpers.config import MAX_LEN
    data.x_train = new_train.iloc[:, -MAX_LEN:]
    data.x_val = new_val.iloc[:, -MAX_LEN:]
    data.train_sentences = new_train["text"]
    data.val_sentences = new_val["text"]
    return data
