# This is an early model not used in the final program.
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import LSTM
import matplotlib as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
RANDOM_SEED = 42


df = pd.read_csv("../data/data_encoded.csv")

df["label"].replace({"hate": 1, "nothate": 0}, inplace=True)

train_set = df.loc[(df['split'] == "train") | (df['split'] == "dev")]
x_train = train_set.iloc[:, 13:]
y_train = OneHotEncoder(sparse=False).fit_transform(
  train_set["label"].to_numpy().reshape(-1, 1)
)

test_set = df.loc[df['split'] == "test"]
x_test = test_set.iloc[:, 13:]
y_test = OneHotEncoder(sparse=False).fit_transform(
  test_set["label"].to_numpy().reshape(-1, 1)
)

# RESET INDICES
x_test = x_test.reset_index(drop=True)
x_train = x_train.reset_index(drop=True)
test_set = test_set.reset_index(drop=True)

# MODEL

model = keras.Sequential()

model.add(
  keras.layers.Dense(
    units=256,
    input_shape=(x_train.shape[1], ),
    activation='relu'
  )
)
model.add(
  keras.layers.Dropout(rate=0.2)
)

model.add(
  keras.layers.Bidirectional(LSTM(64, return_sequences=True)
))

model.add(
  keras.layers.Bidirectional(LSTM(32)
))

model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(2, activation='sigmoid'))
model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

print(model.summary())
print("training")
history = model.fit(
    x_train, y_train,
    epochs=15,
    validation_split=0.1,
    verbose=1,
    shuffle=True
)

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("Cross-entropy loss")
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

probabilities = model.predict(x_test)
predictions = np.argmax(probabilities, axis=1)
target = np.argmax(y_test, axis=1)

print(confusion_matrix(target, predictions))
print(accuracy_score(target, predictions))

for i in range(len(predictions)):
    print("Target: ", target[i], " , prediction: ", predictions[i], " || ", test_set["text"][i])

print("end")