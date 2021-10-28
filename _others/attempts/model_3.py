from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from tensorflow import keras
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
RANDOM_SEED = 42

print("Reading...")
df = pd.read_csv("/data/data_preprocessed_with_stopwords.csv")
print("Reading done.")


df["label"].replace({"hate": 1, "nothate": 0}, inplace=True)

train_set = df.loc[(df['split'] == "train")]
x_train = train_set["text"]
y_train = OneHotEncoder(sparse=False).fit_transform(
  train_set["label"].to_numpy().reshape(-1, 1)
)

val_set = df.loc[(df['split'] == "dev")]
x_val = val_set["text"]
y_val = OneHotEncoder(sparse=False).fit_transform(
  val_set["label"].to_numpy().reshape(-1, 1)
)

test_set = df.loc[df['split'] == "test"]
x_test = test_set["text"]
y_test = OneHotEncoder(sparse=False).fit_transform(
  test_set["label"].to_numpy().reshape(-1, 1)
)

oov_token = '<unk>'
padding_type = 'post'
trunc_type = 'post'
embedding_dim = 100
max_len = max([len(x) for x in x_train])
max_len = 70

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index
vocab_size = len(word_index)
print('Vocab size : ', vocab_size)

train_seq = tokenizer.texts_to_sequences(x_train)
train_pad = pad_sequences(train_seq, padding=padding_type, truncating=trunc_type, maxlen=max_len)

valid_seq = tokenizer.texts_to_sequences(x_val)
valid_pad = pad_sequences(valid_seq, padding=padding_type, truncating=trunc_type, maxlen=max_len)

test_seq = tokenizer.texts_to_sequences(x_test)
test_pad = pad_sequences(test_seq, padding=padding_type, truncating=trunc_type, maxlen=max_len)

embeddings_idx = {}
glove_path = '/data/glove.6B.100d.txt'
with open(glove_path, 'r', encoding="utf8") as f:
    print("Loading words...")
    for line in f:
        data = line.split()
        word = data[0]
        values = np.asarray(data[1:], dtype=np.float32)
        embeddings_idx[word] = values
    print("Loading words done.")


print('Building Embedding Matrix...')
embeddings_matrix = np.zeros((vocab_size + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vec = embeddings_idx.get(word)
    if embedding_vec is not None:
        embeddings_matrix[i] = embedding_vec
print('Embedding Matrix Generating...')
print('Embedding Matrix Shape -> ', embeddings_matrix.shape)

model = keras.models.Sequential([
    Embedding(vocab_size + 1, embedding_dim, input_length=max_len, weights=[embeddings_matrix], trainable=False),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.3)),
    Bidirectional(LSTM(32, dropout=0.3)),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model.summary()

optimizer = Adam(learning_rate=0.005)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(train_pad, y_train, epochs=10, validation_data=(valid_pad, y_val), batch_size=64)

import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel('# epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

probabilities = model.predict(test_pad)
predictions = np.argmax(probabilities, axis=1)
target = np.argmax(y_test, axis=1)

print(confusion_matrix(target, predictions))
print(accuracy_score(target, predictions))
