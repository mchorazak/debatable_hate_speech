from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def print_scores(model):
    print("Scores for ", model.name, ":")
    print(confusion_matrix(model.data.y_test["label"], model.predictions))
    print(accuracy_score(model.data.y_test["label"], model.predictions))
    print(classification_report(model.data.y_test["label"], model.predictions))


def print_scores1(target, predictions):
    print(confusion_matrix(target, predictions))
    print("Accuracy: ", accuracy_score(target, predictions))
    # print(classification_report(target, predictions))


def disputed(model_1, model_2):
    indices = []
    counter = 0
    length = len(model_1.predictions)
    for i in range(length):
        if model_1.predictions[i] != model_2.predictions[i]:
            if model_1.predictions[i] > model_2.predictions[i]:
                counter+=1
            # print(data.test_sentences[i], "\n")
            indices.append(i)
    print("From ", length, " test sentences, ", len(indices), " are disputed.")
    return indices
