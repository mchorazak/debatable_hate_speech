import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("../data/Dynamically Generated Hate Dataset v0.2.3.csv")
data = df

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers

max_words = 5000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data["text"])
sequences = tokenizer.texts_to_sequences(data["text"])
tweets = pad_sequences(sequences, maxlen=max_len)
print(tweets)


df["label"].replace({"hate": 1, "nothate": 0}, inplace=True)
text = pd.DataFrame(tweets)
new = pd.concat([df, text], axis=1)

train_set = new.loc[(df['split'] == "train")]
x_train = train_set.iloc[:, 13:]
y_train = OneHotEncoder(sparse=False).fit_transform(
  train_set["label"].to_numpy().reshape(-1, 1)
)

val_set = new.loc[(df['split'] == "dev")]
x_val = val_set.iloc[:, 13:]
y_val = OneHotEncoder(sparse=False).fit_transform(
  val_set["label"].to_numpy().reshape(-1, 1)
)

test_set = new.loc[df['split'] == "test"]
x_test = test_set.iloc[:, 13:]
y_test = OneHotEncoder(sparse=False).fit_transform(
  test_set["label"].to_numpy().reshape(-1, 1)
)
# many padding zeros at this point. Reject long tweets?

from keras.layers import Embedding
embedding_layer = Embedding(1000, 64)

from keras.models import Sequential
from keras import layers
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint

model1 = Sequential()
model1.add(layers.Embedding(max_words, 20)) #The embedding layer
model1.add(layers.LSTM(15, dropout=0.5)) #Our LSTM layer
model1.add(layers.Dense(2, activation='softmax'))

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpoint1 = ModelCheckpoint("best_model1.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto',
                              period=1, save_weights_only=False)
history = model1.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[checkpoint1])

probabilities = model1.predict(x_test)
predictions = np.argmax(probabilities, axis=1)
target = np.argmax(y_test, axis=1)

print(confusion_matrix(target, predictions))
print(accuracy_score(target, predictions))

print("end.")
