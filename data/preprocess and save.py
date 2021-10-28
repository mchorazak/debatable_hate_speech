import pandas as pd
from text_preprocessing import text_preprocessing

print("Reading...")
df = pd.read_csv("data/Dynamically Generated Hate Dataset v0.2.3_original.csv")
print("Reading done.")

# df = df.sample(frac=0.1)
# df = df.reset_index(drop=True)

print("Preprocessing...")
for i in range(len(df["text"])):
    df["text"][i] = text_preprocessing(df["text"][i], stop_words=False)
print("Preprocessing done.")

df.to_csv('data_preprocessed_with_stopwords_original.csv', index=False)
print("end")
