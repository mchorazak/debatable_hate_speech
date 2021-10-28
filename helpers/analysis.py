from helpers.data_loader import get_data
import pandas as pd
from helpers.printing import print_scores1

# load both data and predictions and prepare for concatination
data = pd.read_csv("data/Dynamically Generated Hate Dataset v0.2.3.csv")
predictions = pd.read_csv("test_classifications.csv", index_col=False, header=None)
predictionsTransposed = predictions.transpose()  # rows to columns
headers = ["lstm_pred", "bdir_pred", "optimal", "suboptimal"]
predictionsTransposed.columns = headers
data = data.loc[data['split'] == "test"]
data["label"].replace({"hate": 1, "nothate": 0}, inplace=True)
data = data.reset_index(drop=True)

# attach classification columns to the original dataframe of test sentences
test = pd.concat([data, predictionsTransposed], axis=1)

# print scores for three types of classifications
print("LSTM predictions:")
print_scores1(test['label'], test["lstm_pred"])
print("BDIR predictions:")
print_scores1(test['label'], test["bdir_pred"])
print("OPTIMAL discussion predictions:")
print_scores1(test['label'], test["optimal"])
print("SUB-OPTIMAL discussion predictions:")
print_scores1(test['label'], test["suboptimal"])

count = 0
for x in range(len(test)):
    if predictionsTransposed["lstm_pred"][x] == 0 and predictionsTransposed["bdir_pred"][x] == 0:
        count += 1
        print(test["text"][x])

black = test.loc[test['target'] == "bla"]
print("black:")
print_scores1(black["label"], black["discussion"])


woman = test.loc[test['target'] == "wom"]
print("woman:")
print_scores1(woman["label"], woman["discussion"])

print("end")
