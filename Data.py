# Container class for x and y data for each train, val and test set.
# Additionally carries sentences in textual form that are used for prining to the screen.
class Data:
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test,
                 test_sentences, train_sentences, val_sentences):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.test_sentences = test_sentences
        self.train_sentences = train_sentences
        self.val_sentences = val_sentences
