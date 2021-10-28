from debater_base import DebaterBase
import tensorflow as tf


class LSTMDebater(DebaterBase):
    def __init__(self, name, data):
        DebaterBase.__init__(self, name, data)

    # run the training function until MAX_EPOCHS or until early stopping terminates.
    # early stopping terminates when val_accuracy does not improve for three epochs
    # during training, the model with best val_loss is saved to a file.
    # when the training terminates, the model is loaded from file to program for predictions.
    def run(self):
        from helpers.config import MAX_EPOCHS, BATCH_SIZE, PLOTS, CLASS_WEIGHTS
        model = self.define_model()

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
        from keras.callbacks import ModelCheckpoint
        path = "bestLSTM.hdf5"
        checkpoint = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='auto', save_weights_only=False)
        history = model.fit(self.data.x_train, self.data.y_train, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                            validation_data=(self.data.x_val, self.data.y_val), callbacks=[callback, checkpoint],
                            class_weight=CLASS_WEIGHTS)
        from helpers.printing import print_scores1
        if PLOTS:
            self.show_history(history)
        self.probabilities = model.predict(self.data.x_test)[:, 0]
        self.predictions = self.probabilities.round()
        print_scores1(self.data.y_test, self.predictions)

        return max(history.history['val_accuracy'])  # return used for overnight tests

    # Define the structure of the model.
    # Multiple configurations were attempted, including larger number of layers but it did not improve the results.
    @staticmethod
    def define_model():
        from helpers import config as cf
        from keras.models import Sequential
        from keras import layers
        from keras.optimizer_v2.adam import Adam

        model = Sequential()
        model.add(layers.Embedding(cf.MAX_WORDS, cf.LAYER_SIZE, input_length=cf.MAX_LEN))  # embedding layer
        model.add(layers.LSTM(cf.LAYER_SIZE, dropout=cf.DROPOUT))  # the LSTM layer
        model.add(layers.Dense(1, activation='sigmoid'))
        optimizer = Adam(learning_rate=cf.LEARNING_RATE, name='Adam')

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model
